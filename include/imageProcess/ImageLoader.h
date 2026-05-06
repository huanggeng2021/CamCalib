#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace camcalib {    

    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths);

    std::vector<cv::Mat> loadImages();

}