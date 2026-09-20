#include "utils/Config.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace fs = std::filesystem;

namespace camcalib {
namespace {

void readBool(const cv::FileNode& node, bool& value) {
    if (node.empty()) {
        return;
    }
    int rawValue = value ? 1 : 0;
    node >> rawValue;
    value = (rawValue != 0);
}

bool validateConfig(const CalibrationPipelineConfig& config) {
    if (config.dataset.imageDirectory.empty()) {
        std::cerr << "Invalid config: dataset.image_directory must not be empty." << std::endl;
        return false;
    }
    if (config.board.rows <= 0 || config.board.cols <= 0 || config.board.spacingMm <= 0.0) {
        std::cerr << "Invalid config: board rows, cols and spacing_mm must be positive." << std::endl;
        return false;
    }
    if (config.detector.minContourPoints <= 0 ||
        config.detector.minContourArea < 0.0 ||
        config.detector.maxContourArea <= config.detector.minContourArea ||
        config.detector.maxAxisRatio <= 0.0) {
        std::cerr << "Invalid config: detector contour settings or area range." << std::endl;
        return false;
    }
    if (config.detector.markerCount <= 0 || config.detector.markerSpacing <= 0.0 ||
        config.detector.rowTolerance <= 0.0) {
        std::cerr << "Invalid config: detector marker settings must be positive." << std::endl;
        return false;
    }
    if (config.projector.enabled &&
        (config.projector.calibrationDataDirectory.empty() ||
         config.projector.method != "pseudo_camera" ||
         config.projector.phaseFrequencies[0] <= config.projector.phaseFrequencies[1] ||
         config.projector.phaseFrequencies[1] <= config.projector.phaseFrequencies[2] ||
         config.projector.phaseFrequencies[2] <= 0.0f ||
         config.projector.width <= 0 ||
         config.projector.height <= 0 ||
         config.projector.minValidViews < 3)) {
        std::cerr << "Invalid config: projector settings." << std::endl;
        return false;
    }
    return true;
}

}  // namespace

bool ConfigReader::readConfig(
    const std::string& yamlPath,
    CalibrationPipelineConfig& config
) {
    cv::FileStorage fileStorage(yamlPath, cv::FileStorage::READ);
    if (!fileStorage.isOpened()) {
        std::cerr << "Failed to open config file: " << yamlPath << std::endl;
        return false;
    }

    const cv::FileNode datasetNode = fileStorage["dataset"];
    datasetNode["image_directory"] >> config.dataset.imageDirectory;

    config.dataset.imageExtensions.clear();
    const cv::FileNode extNode = datasetNode["image_extensions"];
    if (extNode.type() == cv::FileNode::SEQ) {
        for (const auto& ext : extNode) {
            config.dataset.imageExtensions.push_back(static_cast<std::string>(ext));
        }
    }
    readBool(datasetNode["read_grayscale"], config.dataset.readGrayscale);

    const cv::FileNode boardNode = fileStorage["board"];
    boardNode["rows"] >> config.board.rows;
    boardNode["cols"] >> config.board.cols;
    boardNode["spacing_mm"] >> config.board.spacingMm;

    const cv::FileNode detectorNode = fileStorage["detector"];
    detectorNode["min_contour_points"] >> config.detector.minContourPoints;
    detectorNode["min_contour_area"] >> config.detector.minContourArea;
    detectorNode["max_contour_area"] >> config.detector.maxContourArea;
    detectorNode["max_axis_ratio"] >> config.detector.maxAxisRatio;
    detectorNode["marker_count"] >> config.detector.markerCount;
    detectorNode["marker_spacing"] >> config.detector.markerSpacing;
    detectorNode["row_tolerance"] >> config.detector.rowTolerance;
    readBool(detectorNode["enable_subpixel"], config.detector.enableSubpixel);
    readBool(
        detectorNode["black_circles_on_white_background"],
        config.detector.blackCirclesOnWhiteBackground
    );

    const cv::FileNode projectorNode = fileStorage["projector"];
    readBool(projectorNode["enabled"], config.projector.enabled);
    readBool(projectorNode["use_cuda"], config.projector.useCuda);
    if (!projectorNode["method"].empty()) {
        projectorNode["method"] >> config.projector.method;
    }
    if (!projectorNode["calibration_data_directory"].empty()) {
        projectorNode["calibration_data_directory"] >>
            config.projector.calibrationDataDirectory;
    }
    const cv::FileNode phaseFrequenciesNode = projectorNode["phase_frequencies"];
    if (phaseFrequenciesNode.type() == cv::FileNode::SEQ &&
        phaseFrequenciesNode.size() == config.projector.phaseFrequencies.size()) {
        size_t frequencyIndex = 0;
        for (const cv::FileNode& frequencyNode : phaseFrequenciesNode) {
            config.projector.phaseFrequencies[frequencyIndex++] =
                static_cast<float>(frequencyNode.real());
        }
    }
    if (!projectorNode["width"].empty()) {
        projectorNode["width"] >> config.projector.width;
    }
    if (!projectorNode["height"].empty()) {
        projectorNode["height"] >> config.projector.height;
    }
    if (!projectorNode["min_valid_views"].empty()) {
        projectorNode["min_valid_views"] >> config.projector.minValidViews;
    }

    const cv::FileNode loggingNode = fileStorage["logging"];
    readBool(loggingNode["enabled"], config.logging.enabled);
    if (!loggingNode["output_file"].empty()) {
        loggingNode["output_file"] >> config.logging.outputFile;
    }

    const cv::FileNode debugNode = fileStorage["debug"];
    readBool(debugNode["enabled"], config.debug.enabled);
    readBool(debugNode["save_images"], config.debug.saveImages);
    readBool(debugNode["show_windows"], config.debug.showWindows);
    if (!debugNode["output_directory"].empty()) {
        debugNode["output_directory"] >> config.debug.outputDirectory;
    }

    fileStorage.release();
    return validateConfig(config);
}

std::vector<std::string> ConfigReader::getImageFiles(
    const std::string& dir,
    const std::vector<std::string>& extensions
) {
    std::vector<std::string> files;

    std::vector<std::string> normalizedExtensions = extensions;
    for (auto& ext : normalizedExtensions) {
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    }

    if (normalizedExtensions.empty()) {
        normalizedExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    }

    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });

                if (std::find(normalizedExtensions.begin(), normalizedExtensions.end(), ext) != normalizedExtensions.end()) {
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
