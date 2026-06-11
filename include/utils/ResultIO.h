#pragma once

#include "core/CalibrationData.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <type_traits>

namespace camcalib::utils {

bool prepareDebugOutputDirectory(const std::filesystem::path& debugRoot);

bool saveDebugImage(const std::filesystem::path& outputPath, const cv::Mat& image);

template<typename PointT>
bool saveEdgesToText(
    const std::filesystem::path& outputPath,
    const std::vector<std::vector<PointT>>& edges
){
    try{
        std::filesystem::create_directories(outputPath.parent_path());
    }catch(const std::filesystem::filesystem_error&){
        return false;
    }

    std::ofstream outFile(outputPath);
    if(!outFile.is_open()){
        return false;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "# contour_index point_index x y\n";
    for(size_t contourIndex = 0; contourIndex < edges.size(); ++contourIndex){
        for(size_t pointIndex = 0; pointIndex < edges[contourIndex].size(); ++pointIndex){
            outFile
                << contourIndex << ' '
                << pointIndex << ' '
                << static_cast<double>(edges[contourIndex][pointIndex].x) << ' '
                << static_cast<double>(edges[contourIndex][pointIndex].y) << '\n';
        }
    }

    return true;
}

void drawCenterPoints(cv::Mat& image, const CircleSet& centerPoints);

template<typename PointT>
cv::Mat renderEdgeAndCircleCenters(
    const cv::Mat& image,
    const std::vector<std::vector<PointT>>& edges,
    const CircleSet& centerPoints
){
    cv::Mat display;
    if(image.channels() == 1){
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    }else{
        display = image.clone();
    }

    cv::Mat overlay = display.clone();
    constexpr bool isSubPixel = std::is_floating_point_v<typename PointT::value_type>;
    constexpr int pointShift = isSubPixel ? 8 : 0;
    constexpr int pointScale = 1 << pointShift;

    for(size_t contourIndex = 0; contourIndex < edges.size(); ++contourIndex){
        if(edges[contourIndex].empty()){
            continue;
        }

        cv::Point2d meanPoint(0.0, 0.0);
        std::vector<cv::Point> polyline;
        polyline.reserve(edges[contourIndex].size());

        for(const PointT& point : edges[contourIndex]){
            meanPoint += cv::Point2d(static_cast<double>(point.x), static_cast<double>(point.y));
            polyline.emplace_back(
                static_cast<int>(std::round(static_cast<double>(point.x) * pointScale)),
                static_cast<int>(std::round(static_cast<double>(point.y) * pointScale))
            );
        }

        for(size_t pointIndex = 1; pointIndex < polyline.size(); ++pointIndex){
            cv::line(
                overlay,
                polyline[pointIndex - 1],
                polyline[pointIndex],
                cv::Scalar(0, 0, 255),
                1,
                cv::LINE_AA,
                pointShift
            );
        }

        if(polyline.size() > 2){
            cv::line(
                overlay,
                polyline.back(),
                polyline.front(),
                cv::Scalar(0, 0, 255),
                1,
                cv::LINE_AA,
                pointShift
            );
        }

        const size_t markerStep = std::max<size_t>(1, edges[contourIndex].size() / 24);
        for(size_t pointIndex = 0; pointIndex < edges[contourIndex].size(); pointIndex += markerStep){
            const PointT& point = edges[contourIndex][pointIndex];
            cv::circle(
                overlay,
                cv::Point(
                    static_cast<int>(std::round(static_cast<double>(point.x))),
                    static_cast<int>(std::round(static_cast<double>(point.y)))
                ),
                1,
                cv::Scalar(80, 220, 255),
                -1,
                cv::LINE_AA
            );
        }

        meanPoint *= (1.0 / static_cast<double>(edges[contourIndex].size()));
        cv::putText(
            overlay,
            std::to_string(contourIndex),
            cv::Point(
                static_cast<int>(std::round(meanPoint.x)),
                std::max(15, static_cast<int>(std::round(meanPoint.y)) - 5)
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 255, 0),
            1,
            cv::LINE_AA
        );
    }

    cv::addWeighted(overlay, 0.75, display, 0.25, 0.0, display);
    drawCenterPoints(display, centerPoints);
    return display;
}

cv::Mat renderSortedCircleCenters(
    const cv::Mat& image,
    const CircleSet& sortedCenterPoints
);

void showSortedCircleCenters(
    const cv::Mat& image,
    const CircleSet& sortedCenterPoints,
    const std::string& windowName
);

void showWarpedImage(
    const cv::Mat& image,
    const Eigen::Matrix3d& homography,
    const std::string& windowName
);

}  // namespace camcalib::utils
