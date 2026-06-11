#pragma once
#include <opencv2/core.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "utils/Config.h"

namespace camcalib {    

class ImageProcess{

public:

    // 加载图像入口
    std::vector<cv::Mat> loadImages(const CaliConfig& config);

    // 标定入口 
    void runCalibrate();
       

private:
    bool log_enabled_ = false;
    std::ofstream log_stream_;

    // 加载灰度图像
    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths);

    void initializeLogger(const CaliConfig& config);
    void shutdownLogger();
    void logMessage(const std::string& level, const std::string& message);
    void logInfo(const std::string& message);
    void logError(const std::string& message);
};

    

}
