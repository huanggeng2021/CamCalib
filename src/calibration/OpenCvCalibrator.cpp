#include "calibration/OpenCvCalibrator.h"

#include <opencv2/calib3d.hpp>
#include <utility>

namespace camcalib {

OpenCvCalibrator::OpenCvCalibrator(BoardConfig boardConfig)
{
    static_cast<void>(boardConfig);
}

CalibrationResult OpenCvCalibrator::calibrate(
    const CalibrationDataset& dataset,
    const DetectionResult& detection
) const {
    CalibrationResult result;
    result.solverName = "cv::calibrateCamera";

    if(dataset.empty()){
        return result;
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    for(const ViewObservation& view : detection.views){
        if(!view.valid ||
           view.objectPoints.empty() ||
           view.objectPoints.size() != view.imagePoints.size()){
            continue;
        }

        objectPoints.push_back(view.objectPoints);

        std::vector<cv::Point2f> imageViewPoints;
        imageViewPoints.reserve(view.imagePoints.size());
        for(const cv::Point2d& point : view.imagePoints){
            imageViewPoints.emplace_back(point);
        }
        imagePoints.push_back(std::move(imageViewPoints));
    }

    if(objectPoints.empty() || imagePoints.empty()){
        return result;
    }

    result.globalRmse = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        dataset.imageSize,
        result.cameraMatrix,
        result.distCoeffs,
        result.rotationVectors,
        result.translationVectors,
        cv::CALIB_FIX_K3
    );
    result.converged = !result.cameraMatrix.empty() && !result.distCoeffs.empty();
    return result;
}

}  // namespace camcalib
