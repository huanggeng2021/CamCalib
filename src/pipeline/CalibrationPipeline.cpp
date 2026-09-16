#include "pipeline/CalibrationPipeline.h"

#include "calibration/CameraProjectorCalibrator.h"
#include "calibration/OpenCvCalibrator.h"
#include "dataset/DatasetLoader.h"
#include "dataset/ProjectorDatasetLoader.h"
#include "detection/CircleGridDetector.h"
#include "evaluation/ReprojectionEvaluator.h"
#include "imageProcess/ThreeFrequencyFourStepPhase.h"
#include "projector/ProjectorPointMatcher.h"
#include "utils/Config.h"
#include "utils/Logger.h"
#include "utils/ResultIO.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace camcalib {
namespace {

void logDetection(const DetectionResult& detection){
    for(size_t imageIndex = 0; imageIndex < detection.views.size(); ++imageIndex){
        const ViewObservation& view = detection.views[imageIndex];
        if(view.valid){
            utils::logInfo(
                "Image " + std::to_string(imageIndex) +
                ": points=" + std::to_string(view.imagePoints.size()) +
                ", score=" + std::to_string(view.detectionScore)
            );
        }else{
            utils::logError(
                "Image " + std::to_string(imageIndex) +
                " detection failed: " + view.failureReason
            );
        }
    }
}

void logEvaluation(const EvaluationReport& evaluation){
    for(const ViewError& viewError : evaluation.views){
        std::ostringstream message;
        message << "Image " << viewError.viewIndex << " reprojection error = "
                << std::fixed << std::setprecision(6) << viewError.rmse
                << " pixels";
        utils::logInfo(message.str());
        std::cout << message.str() << std::endl;
    }
}

void logProjectorPoints(const std::vector<ProjectorPoseData>& poses){
    for(const ProjectorPoseData& pose : poses){
        size_t validPointCount = 0;
        for(const cv::Point2d& point : pose.projectorPoints){
            if(std::isfinite(point.x) && std::isfinite(point.y)){
                ++validPointCount;
            }
        }
        utils::logInfo(
            pose.poseName +
            " projector points=" + std::to_string(validPointCount)
        );
    }
}

}  // namespace

CalibrationResult CalibrationPipeline::calibrateAndEvaluate(
    const CalibrationDataset& dataset,
    DetectionResult& detection,
    const BoardConfig& board,
    EvaluationReport& evaluation
) const {
    //OpenCvCalibrator calibrator(board);
    //CalibrationResult calibration = calibrator.calibrate(dataset, detection);

    CustomCalibrator cucalibrator(board);
    CalibrationResult calibration = cucalibrator.calibrate(dataset, detection);
    if(!calibration.converged){
        return calibration;
    }

    ReprojectionEvaluator evaluator;
    evaluation = evaluator.evaluate(detection, calibration);
    return calibration;
}

DetectionResult CalibrationPipeline::buildProjectorDetection(
    const std::vector<ViewObservation>& cameraViews,
    const std::vector<ProjectorPoseData>& poses,
    cv::Size projectorSize
) const {
    DetectionResult detection;
    const size_t viewCount = std::min(cameraViews.size(), poses.size());
    detection.views.reserve(viewCount);

    for(size_t viewIndex = 0; viewIndex < viewCount; ++viewIndex){
        const ViewObservation& cameraView = cameraViews[viewIndex];
        const ProjectorPoseData& pose = poses[viewIndex];

        ViewObservation projectorView;
        projectorView.imagePath = pose.poseName;
        projectorView.imageSize = projectorSize;

        const size_t pointCount = std::min(
            cameraView.objectPoints.size(),
            pose.projectorPoints.size()
        );
        for(size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex){
            const cv::Point2d& projectorPoint =
                pose.projectorPoints[pointIndex];
            if(!std::isfinite(projectorPoint.x) ||
               !std::isfinite(projectorPoint.y)){
                continue;
            }

            projectorView.objectPoints.push_back(
                cameraView.objectPoints[pointIndex]
            );
            projectorView.imagePoints.push_back(projectorPoint);
        }

        projectorView.detectionScore = pointCount == 0
            ? 0.0
            : static_cast<double>(projectorView.imagePoints.size()) /
              static_cast<double>(pointCount);
        projectorView.valid = cameraView.valid &&
            projectorView.imagePoints.size() >= 4;
        if(!projectorView.valid){
            projectorView.failureReason =
                "Too few valid projector calibration points.";
        }
        detection.views.push_back(std::move(projectorView));
    }

    return detection;
}

bool CalibrationPipeline::solveProjectorPhases(
    std::vector<ProjectorPoseData>& poses,
    const std::array<float, 3>& frequencies,
    int minValidViews
) const {
    int validViewCount = 0;

    for(ProjectorPoseData& pose : poses){
        try{
            pose.xAbsolutePhase = ThreeFrequencyFourStepPhase::solve(
                pose.xImages,
                frequencies
            ).unwrappedPhase;
            pose.yAbsolutePhase = ThreeFrequencyFourStepPhase::solve(
                pose.yImages,
                frequencies
            ).unwrappedPhase;
        }catch(const cv::Exception& exception){
            pose.xAbsolutePhase.release();
            pose.yAbsolutePhase.release();
            utils::logError(
                "Phase solving failed for " + pose.poseName +
                ": " + exception.what()
            );
            continue;
        }

        if(pose.xAbsolutePhase.empty() ||
           pose.yAbsolutePhase.empty() ||
           pose.xAbsolutePhase.size() != pose.yAbsolutePhase.size()){
            pose.xAbsolutePhase.release();
            pose.yAbsolutePhase.release();
            utils::logError("Invalid absolute phase result for " + pose.poseName);
            continue;
        }

        ++validViewCount;
        utils::logInfo("Absolute phase solved for " + pose.poseName);
    }

    return validViewCount >= minValidViews;
}

bool CalibrationPipeline::runCameraCalibration(
    const std::string& configPath,                
    DetectionResult& cameraDetection,     
    CalibrationResult& cameraCalibration
) const {
    CalibrationPipelineConfig config;
    if(!ConfigReader::readConfig(configPath, config)){
        std::cerr << "Failed to read config file: " << configPath << std::endl;
        return false;
    }

    utils::initializeLogger(config.logging);
    utils::logInfo("Camera calibration started.");
    utils::logInfo("Current working directory: " + std::filesystem::current_path().string());
    utils::logInfo("Loaded config file: " + configPath);

    const std::filesystem::path debugRoot = config.debug.outputDirectory;
    if(config.debug.saveImages && !utils::prepareDebugOutputDirectory(debugRoot)){
        utils::logError("Failed to create debug output directory: " + debugRoot.string());
        utils::shutdownLogger();
        return false;
    }

    DatasetLoader datasetLoader(config.dataset);
    const CalibrationDataset dataset = datasetLoader.load();
    if(dataset.empty()){
        utils::logError("Calibration aborted because no input images were loaded.");
        utils::shutdownLogger();
        return false;
    }

    CircleGridDetector detector(config.board, config.detector);
    cameraDetection = detector.detect(dataset);
    if(config.debug.saveImages &&
       !utils::saveDetectionDebugResults(debugRoot, dataset, cameraDetection)){
        utils::logError("Some detection debug results could not be saved.");
    }
    if(config.debug.enabled && config.debug.showWindows){
        utils::showDetectionDebugResults(dataset, cameraDetection);
    }
    logDetection(cameraDetection);

    EvaluationReport evaluation;
    cameraCalibration = calibrateAndEvaluate(
        dataset,
        cameraDetection,
        config.board,
        evaluation
    );
    if(!cameraCalibration.converged){
        utils::logError("Calibration failed because no valid observations were produced.");
        utils::shutdownLogger();
        return false;
    }

    if(!utils::saveCalibrationResult(
           std::filesystem::path(configPath).parent_path() /
               "camera_calibration.yaml",
           "camera",
           cameraCalibration)){
        utils::logError("Failed to save camera calibration result.");
    }
    utils::logInfo(
        "Calibration finished. RMS = " +
        std::to_string(cameraCalibration.globalRmse)
    );
    logEvaluation(evaluation);
    utils::shutdownLogger();
    return true;
}

bool CalibrationPipeline::runProjectorCalibration(
    const std::string& configPath,
    const std::vector<ViewObservation>& cameraViews,
    std::vector<ProjectorPoseData>& projectorPoses,
    CalibrationResult& projectorCalibration
) const {
    CalibrationPipelineConfig config;
    if(!ConfigReader::readConfig(configPath, config)){
        std::cerr << "Failed to read config file: " << configPath << std::endl;
        return false;
    }
    if(!config.projector.enabled){
        std::cout << "Projector calibration is disabled." << std::endl;
        return false;
    }

    utils::initializeLogger(config.logging);
    utils::logInfo("Projector calibration started.");

    const std::filesystem::path debugRoot = config.debug.outputDirectory;
    if(config.debug.saveImages && !utils::prepareDebugOutputDirectory(debugRoot)){
        utils::logError("Failed to create debug output directory: " + debugRoot.string());
        utils::shutdownLogger();
        return false;
    }

    ProjectorDatasetLoader datasetLoader(
        config.projector,
        config.dataset.imageExtensions
    );
    projectorPoses = datasetLoader.load();
    if(projectorPoses.empty()){
        utils::logError("No valid projector calibration poses were loaded.");
        utils::shutdownLogger();
        return false;
    }

    utils::logInfo(
        "Loaded projector calibration poses: " +
        std::to_string(projectorPoses.size())
    );

    if(!solveProjectorPhases(
           projectorPoses,
           config.projector.phaseFrequencies,
           config.projector.minValidViews)){
        utils::logError("Too few valid projector phase results.");
        utils::shutdownLogger();
        return false;
    }

    utils::logInfo("Projector absolute phase solving finished.");
    if(config.debug.saveImages &&
       !utils::saveProjectorPhaseDebugResults(debugRoot, projectorPoses)){
        utils::logError("Some projector phase debug images could not be saved.");
    }

    if(cameraViews.size() != projectorPoses.size()){
        utils::logError("Camera detections and projector poses do not match.");
        utils::shutdownLogger();
        return false;
    }

    ProjectorPointMatcher matcher;
    matcher.match(
        cameraViews,
        projectorPoses,
        config.projector
    );
    logProjectorPoints(projectorPoses);

    utils::logInfo("Projector feature coordinates calculated.");

    const cv::Size projectorSize(
        config.projector.width,
        config.projector.height
    );
    DetectionResult projectorDetection = buildProjectorDetection(
        cameraViews,
        projectorPoses,
        projectorSize
    );

    CalibrationDataset projectorDataset;
    projectorDataset.imageSize = projectorSize;
    projectorDataset.images.reserve(projectorPoses.size());
    for(const ProjectorPoseData& pose : projectorPoses){
        if(pose.xImages.empty()){
            continue;
        }

        DatasetImage image;
        image.path = pose.poseName;
        image.image = pose.xImages.front();
        projectorDataset.images.push_back(std::move(image));
    }

    EvaluationReport evaluation;
    projectorCalibration = calibrateAndEvaluate(
        projectorDataset,
        projectorDetection,
        config.board,
        evaluation
    );
    projectorCalibration.solverName = "pseudo_camera_cv::calibrateCamera";
    if(!projectorCalibration.converged){
        utils::logError("Projector calibration failed.");
        utils::shutdownLogger();
        return false;
    }

    if(!utils::saveCalibrationResult(
           std::filesystem::path(configPath).parent_path() /
               "projector_calibration.yaml",
           "projector",
           projectorCalibration)){
        utils::logError("Failed to save projector calibration result.");
    }
    utils::logInfo(
        "Projector calibration finished. RMS = " +
        std::to_string(projectorCalibration.globalRmse)
    );
    logEvaluation(evaluation);
    utils::shutdownLogger();
    return true;
}

bool CalibrationPipeline::runCameraProjectorCalibration(
    const std::string& configPath,
    const std::vector<ViewObservation>& cameraViews,
    const std::vector<ProjectorPoseData>& projectorPoses,
    const CalibrationResult& cameraCalibration,
    const CalibrationResult& projectorCalibration
) const {
    CalibrationPipelineConfig config;
    if(!ConfigReader::readConfig(configPath, config)){
        std::cerr << "Failed to read config file: " << configPath << std::endl;
        return false;
    }

    utils::initializeLogger(config.logging);
    utils::logInfo("Camera-projector joint calibration started.");

    CameraProjectorCalibrator stereoCalibrator;
    const CameraProjectorCalibrationResult stereoCalibration =
        stereoCalibrator.calibrate(
            cameraViews,
            projectorPoses,
            cameraCalibration,
            projectorCalibration,
            config.projector.minValidViews
        );
    if(!stereoCalibration.converged){
        utils::logError("Camera-projector joint calibration failed.");
        utils::shutdownLogger();
        return false;
    }

    if(!utils::saveCameraProjectorCalibrationResult(
           std::filesystem::path(configPath).parent_path() /
               "camera_projector_calibration.yaml",
           stereoCalibration)){
        utils::logError("Failed to save camera-projector calibration result.");
    }
    utils::logInfo(
        "Camera-projector joint calibration finished. RMS = " +
        std::to_string(stereoCalibration.globalRmse)
    );
    utils::shutdownLogger();
    return true;
}

}  // namespace camcalib
