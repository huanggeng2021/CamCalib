#pragma once

#include "core/CalibrationData.h"
#include <opencv2/core.hpp>
#include <vector>

namespace camcalib::utils {

std::vector<std::vector<cv::Point2f>> collectImagePoints(const CircleSets& sortedBoardCircles);

std::vector<double> calculatePerImageReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rotationVectors,
    const std::vector<cv::Mat>& translationVectors,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs
);

}  // namespace camcalib::utils
