#include "utils/Reprojection.h"

#include <cmath>
#include <opencv2/calib3d.hpp>

namespace camcalib::utils {

std::vector<std::vector<cv::Point2f>> collectImagePoints(const CircleSets& sortedBoardCircles){
    std::vector<std::vector<cv::Point2f>> imagePoints;
    imagePoints.reserve(sortedBoardCircles.size());

    for(const CircleSet& imageCircles : sortedBoardCircles){
        std::vector<cv::Point2f> imageCircleCenters;
        imageCircleCenters.reserve(imageCircles.size());
        for(const Circle& circle : imageCircles){
            imageCircleCenters.emplace_back(circle.center);
        }
        imagePoints.push_back(imageCircleCenters);
    }

    return imagePoints;
}

std::vector<double> calculatePerImageReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rotationVectors,
    const std::vector<cv::Mat>& translationVectors,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs
){
    std::vector<double> perImageErrors;
    perImageErrors.reserve(imagePoints.size());

    for(size_t imageIndex = 0; imageIndex < imagePoints.size(); ++imageIndex){
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(
            objectPoints[imageIndex],
            rotationVectors[imageIndex],
            translationVectors[imageIndex],
            cameraMatrix,
            distCoeffs,
            projectedPoints
        );

        double squaredErrorSum = 0.0;
        for(size_t pointIndex = 0; pointIndex < imagePoints[imageIndex].size(); ++pointIndex){
            const cv::Point2f diff = imagePoints[imageIndex][pointIndex] - projectedPoints[pointIndex];
            squaredErrorSum += static_cast<double>(diff.dot(diff));
        }

        const double rmsError = imagePoints[imageIndex].empty()
            ? 0.0
            : std::sqrt(squaredErrorSum / static_cast<double>(imagePoints[imageIndex].size()));
        perImageErrors.push_back(rmsError);
    }

    return perImageErrors;
}

}  // namespace camcalib::utils
