#pragma once

#include "core/CalibrationData.h"
#include <opencv2/core.hpp>
#include <vector>

namespace camcalib::image {

PixelContoursByImage detectEdges(const std::vector<cv::Mat>& images);

SubPixelContoursByImage detectSubPixelEdges(
    const std::vector<cv::Mat>& images,
    const PixelContoursByImage& pixelEdges
);

}  // namespace camcalib::image
