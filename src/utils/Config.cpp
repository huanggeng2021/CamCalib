#include "utils/Config.h"
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace camcalib {

bool ConfigReader::readConfig(const std::string& yaml_path, CaliConfig& config) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << yaml_path << std::endl;
        return false;
    }
    fs["board_width"] >> config.board_width;
    fs["board_height"] >> config.board_height;
    fs["square_size"] >> config.square_size;
    fs["image_dir"] >> config.image_dir;
    fs.release();
    return true;
}

std::vector<std::string> ConfigReader::getImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 支持常见图像格式
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff") {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }
    return files;
}

}  // namespace camcalib
