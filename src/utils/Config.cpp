#include "utils/Config.h"
#include <opencv2/core.hpp>
#include <algorithm>
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

    auto naturalCompare = [](const std::string& a, const std::string& b) {
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j]))) {
                size_t i0 = i, j0 = j;
                while (i0 < a.size() && std::isdigit(static_cast<unsigned char>(a[i0]))) ++i0;
                while (j0 < b.size() && std::isdigit(static_cast<unsigned char>(b[j0]))) ++j0;

                std::string_view sa(a.c_str() + i, i0 - i);
                std::string_view sb(b.c_str() + j, j0 - j);

                // skip leading zeros for numeric comparison
                size_t k = 0;
                while (k < sa.size() && sa[k] == '0') ++k;
                size_t l = 0;
                while (l < sb.size() && sb[l] == '0') ++l;
                std::string_view sa_trim = sa.substr(k);
                std::string_view sb_trim = sb.substr(l);

                if (sa_trim.size() != sb_trim.size())
                    return sa_trim.size() < sb_trim.size();
                if (sa_trim != sb_trim)
                    return sa_trim < sb_trim;
                if (sa.size() != sb.size())
                    return sa.size() < sb.size();

                i = i0;
                j = j0;
                continue;
            }
            if (a[i] != b[j])
                return a[i] < b[j];
            ++i;
            ++j;
        }
        return a.size() < b.size();
    };

    std::sort(files.begin(), files.end(), naturalCompare);
    return files;
}

}  // namespace camcalib
