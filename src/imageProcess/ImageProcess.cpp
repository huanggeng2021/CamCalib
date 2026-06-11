#include "imageProcess/ImageProcess.h"

#include "core/CalibrationData.h"
#include "imageProcess/EdgeProcessing.h"
#include "solver/ClosedFormSolver.h"
#include "solver/Homography.h"
#include "utils/Config.h"
#include "utils/Reprojection.h"
#include "utils/ResultIO.h"

#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <sstream>

namespace camcalib {

std::vector<cv::Mat> ImageProcess::loadGrayImage(const std::vector<std::string>& imagePaths){
    std::vector<cv::Mat> images;
    images.reserve(imagePaths.size());

    for(const std::string& path : imagePaths){
        cv::Mat image = cv::imread(path);
        if(image.empty()){
            logError("Failed to load image: " + path);
            continue;
        }

        cv::Mat grayImage;
        if(image.channels() == 3){
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        }else if(image.channels() == 1){
            grayImage = image;
        }

        images.push_back(grayImage);
    }

    return images;
}

std::vector<cv::Mat> ImageProcess::loadImages(const CaliConfig& config){
    const std::vector<std::string> imageFiles = camcalib::ConfigReader::getImageFiles(
        config.image_dir,
        config.image_extensions
    );
    if(imageFiles.empty()){
        logError("No image files found in directory: " + config.image_dir);
        return {};
    }

    logInfo("Found " + std::to_string(imageFiles.size()) + " image files in " + config.image_dir);
    for(const std::string& file : imageFiles){
        logInfo("Image file: " + file);
    }

    if(config.read_grayscale){
        return loadGrayImage(imageFiles);
    }

    std::vector<cv::Mat> images;
    images.reserve(imageFiles.size());
    for(const std::string& path : imageFiles){
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if(image.empty()){
            logError("Failed to load image: " + path);
            continue;
        }
        images.push_back(image);
    }

    return images;
}

void ImageProcess::initializeLogger(const CaliConfig& config){
    log_enabled_ = config.log_enabled;
    if(!log_enabled_){
        return;
    }

    const std::filesystem::path logPath = config.log_output_file;
    try{
        if(logPath.has_parent_path()){
            std::filesystem::create_directories(logPath.parent_path());
        }
        log_stream_.open(logPath, std::ios::out | std::ios::app);
    }catch(const std::filesystem::filesystem_error& e){
        std::cerr << "Failed to prepare log file: " << e.what() << std::endl;
        log_enabled_ = false;
        return;
    }

    if(!log_stream_.is_open()){
        std::cerr << "Failed to open log file: " << logPath << std::endl;
        log_enabled_ = false;
    }
}

void ImageProcess::shutdownLogger(){
    if(log_stream_.is_open()){
        log_stream_.flush();
        log_stream_.close();
    }
    log_enabled_ = false;
}

void ImageProcess::logMessage(const std::string& level, const std::string& message){
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::ostringstream stream;
    if(localTime != nullptr){
        stream << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
    }else{
        stream << "unknown-time";
    }
    stream << " [" << level << "] " << message;

    if(level == "ERROR"){
        std::cerr << stream.str() << std::endl;
    }else{
        std::cout << stream.str() << std::endl;
    }

    if(log_enabled_ && log_stream_.is_open()){
        log_stream_ << stream.str() << std::endl;
    }
}

void ImageProcess::logInfo(const std::string& message){
    logMessage("INFO", message);
}

void ImageProcess::logError(const std::string& message){
    logMessage("ERROR", message);
}

void ImageProcess::runCalibrate(){
    const std::string configPath = "config/calib_config.yaml";

    CaliConfig config;
    if(!ConfigReader::readConfig(configPath, config)){
        std::cerr << "Failed to read config file: " << configPath << std::endl;
        return;
    }

    initializeLogger(config);
    logInfo("Calibration started.");
    logInfo("Current working directory: " + std::filesystem::current_path().string());
    logInfo("Loaded config file: " + configPath);

    const bool shouldSaveDebugImages = config.debug_save_images;
    const bool shouldShowDebugWindows = config.debug_mode && config.debug_show_windows;
    logInfo(
        "Options: log_enabled=" + std::to_string(config.log_enabled) +
        ", debug_mode=" + std::to_string(config.debug_mode) +
        ", debug_save_images=" + std::to_string(config.debug_save_images) +
        ", debug_show_windows=" + std::to_string(config.debug_show_windows)
    );

    const std::vector<cv::Mat> images = loadImages(config);
    if(images.empty()){
        logError("Calibration aborted because no input images were loaded.");
        shutdownLogger();
        return;
    }

    const std::filesystem::path debugRoot = config.debug_output_dir;
    if(shouldSaveDebugImages && !utils::prepareDebugOutputDirectory(debugRoot)){
        logError("Failed to create debug output directory: " + debugRoot.string());
        shutdownLogger();
        return;
    }

    const PixelContoursByImage pixelEdges = image::detectEdges(images);
    logInfo("Edge detection completed for " + std::to_string(pixelEdges.size()) + " images.");

    const SubPixelContoursByImage subPixelEdges = image::detectSubPixelEdges(images, pixelEdges);
    logInfo("Subpixel edge refinement completed for " + std::to_string(subPixelEdges.size()) + " images.");

    CircleSets fittedCircleCenters;
    fittedCircleCenters.reserve(subPixelEdges.size());

    for(size_t imageIndex = 0; imageIndex < subPixelEdges.size(); ++imageIndex){
        CircleSet fittedCircles;
        fittedCircles.reserve(subPixelEdges[imageIndex].size());
        for(const SubPixelContour& contour : subPixelEdges[imageIndex]){
            fittedCircles.push_back(solver::fitCircleToEdges(contour));
        }
        fittedCircleCenters.push_back(fittedCircles);

        logInfo(
            "Image " + std::to_string(imageIndex) +
            ": contours=" + std::to_string(pixelEdges[imageIndex].size()) +
            ", fitted_centers=" + std::to_string(fittedCircles.size())
        );

        if(shouldSaveDebugImages){
            std::ostringstream imageFolderName;
            imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
            const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

            if(!utils::saveEdgesToText(imageDebugDir / "00_pixel_edges.txt", pixelEdges[imageIndex])){
                logError("Failed to save pixel edge text for image " + std::to_string(imageIndex));
            }
            if(!utils::saveEdgesToText(imageDebugDir / "01_subpixel_edges.txt", subPixelEdges[imageIndex])){
                logError("Failed to save subpixel edge text for image " + std::to_string(imageIndex));
            }
            if(!utils::saveDebugImage(
                imageDebugDir / "02_detected_edges.png",
                utils::renderEdgeAndCircleCenters(images[imageIndex], subPixelEdges[imageIndex], fittedCircles)
            )){
                logError("Failed to save detected edge image for image " + std::to_string(imageIndex));
            }
            if(!utils::saveDebugImage(
                imageDebugDir / "03_fitted_centers.png",
                utils::renderSortedCircleCenters(images[imageIndex], fittedCircles)
            )){
                logError("Failed to save fitted center image for image " + std::to_string(imageIndex));
            }
        }
    }

    const CircleSets sortedMarkerCenters = solver::sortMarkerCenters(fittedCircleCenters);
    logInfo("Marker sorting completed.");

    if(shouldSaveDebugImages){
        for(size_t imageIndex = 0; imageIndex < sortedMarkerCenters.size() && imageIndex < images.size(); ++imageIndex){
            if(sortedMarkerCenters[imageIndex].empty()){
                logError("Image " + std::to_string(imageIndex) + " has no sorted marker output.");
                continue;
            }

            std::ostringstream imageFolderName;
            imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
            const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();
            if(!utils::saveDebugImage(
                imageDebugDir / "04_sorted_markers.png",
                utils::renderSortedCircleCenters(images[imageIndex], sortedMarkerCenters[imageIndex])
            )){
                logError("Failed to save sorted marker image for image " + std::to_string(imageIndex));
            }
        }
    }

    const std::vector<Eigen::Matrix3d> homographies = solver::findHomography(sortedMarkerCenters);
    logInfo("Homography computation completed.");

    const CircleSets sortedBoardCircles = solver::sortBoardCirclesByHomography(homographies, fittedCircleCenters);
    logInfo("Full board sorting completed.");

    if(shouldSaveDebugImages){
        for(size_t imageIndex = 0; imageIndex < sortedBoardCircles.size() && imageIndex < images.size(); ++imageIndex){
            if(sortedBoardCircles[imageIndex].empty()){
                logError("Image " + std::to_string(imageIndex) + " has no final sorted board output.");
                continue;
            }

            std::ostringstream imageFolderName;
            imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
            const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();
            if(!utils::saveDebugImage(
                imageDebugDir / "05_sorted_board.png",
                utils::renderSortedCircleCenters(images[imageIndex], sortedBoardCircles[imageIndex])
            )){
                logError("Failed to save sorted board image for image " + std::to_string(imageIndex));
            }
        }
    }

    if(shouldShowDebugWindows){
        for(size_t imageIndex = 0; imageIndex < sortedMarkerCenters.size() && imageIndex < images.size(); ++imageIndex){
            if(!sortedMarkerCenters[imageIndex].empty()){
                utils::showSortedCircleCenters(
                    images[imageIndex],
                    sortedMarkerCenters[imageIndex],
                    "Sorted Circle Centers " + std::to_string(imageIndex)
                );
            }
        }

        for(size_t imageIndex = 0; imageIndex < sortedBoardCircles.size() &&
                                   imageIndex < images.size() &&
                                   imageIndex < homographies.size(); ++imageIndex){
            if(sortedBoardCircles[imageIndex].empty()){
                continue;
            }

            utils::showWarpedImage(
                images[imageIndex],
                homographies[imageIndex],
                "Warped Image " + std::to_string(imageIndex)
            );
            utils::showSortedCircleCenters(
                images[imageIndex],
                sortedBoardCircles[imageIndex],
                "Sorted Board Circles " + std::to_string(imageIndex)
            );
        }
    }

    const std::vector<std::vector<cv::Point3f>> worldCoordinates = solver::generateWorldCoordinates(
        static_cast<int>(sortedBoardCircles.size()),
        config.calib_rows,
        config.calib_cols,
        config.calib_centerDistance
    );

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;
    const std::vector<std::vector<cv::Point2f>> imagePoints = utils::collectImagePoints(sortedBoardCircles);

    const double rms = cv::calibrateCamera(
        worldCoordinates,
        imagePoints,
        cv::Size(images[0].cols, images[0].rows),
        cameraMatrix,
        distCoeffs,
        rotationVectors,
        translationVectors
    );
    logInfo("Calibration finished. RMS = " + std::to_string(rms));

    const std::vector<double> perImageErrors = utils::calculatePerImageReprojectionErrors(
        worldCoordinates,
        imagePoints,
        rotationVectors,
        translationVectors,
        cameraMatrix,
        distCoeffs
    );

    for(size_t imageIndex = 0; imageIndex < perImageErrors.size(); ++imageIndex){
        std::ostringstream message;
        message << "Image " << imageIndex << " reprojection error = "
                << std::fixed << std::setprecision(6) << perImageErrors[imageIndex]
                << " pixels";
        logInfo(message.str());
        std::cout << message.str() << std::endl;
    }

    shutdownLogger();
}

}  // namespace camcalib
