#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace camcalib {

struct Circle {
    std::vector<cv::Point2d> edge_points;
    cv::Point2d center;
    double radius = 0.0;
    double board_row = 0.0;
    double board_col = 0.0;
};

using PixelContour = std::vector<cv::Point>;
using PixelContours = std::vector<PixelContour>;
using PixelContoursByImage = std::vector<PixelContours>;

using SubPixelContour = std::vector<cv::Point2d>;
using SubPixelContours = std::vector<SubPixelContour>;
using SubPixelContoursByImage = std::vector<SubPixelContours>;

using CircleSet = std::vector<Circle>;
using CircleSets = std::vector<CircleSet>;

}  // namespace camcalib
