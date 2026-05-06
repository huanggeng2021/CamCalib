#include "imageProcess/ImageLoader.h"
#include "utils/Config.h"
#include <filesystem>
#include <iostream>

namespace camcalib {    

    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths){

        std::vector<cv::Mat> images;
        images.reserve(image_paths.size());  

        for(const auto& path:image_paths){

            cv::Mat image = cv::imread(path);

            if(image.empty()){
                std::cerr << "Failed to load image: " << path << std::endl;
                continue;
            }

            cv::Mat gray;
            if(image.channels() == 3){

                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            
            }else if(image.channels() == 1){

                gray = image;
            }    

            images.push_back(gray);
        }

        return images;
    }



    std::vector<cv::Mat> loadImages(){

        // step1: read config
        std::cout << "Current working directory: " << std::filesystem::current_path().string() << std::endl;
        std::string yaml_path = "config/calib_config.yaml";
        
        camcalib::CaliConfig config;
        if (!camcalib::ConfigReader::readConfig(yaml_path, config)) {
            std::cerr << "Failed to read config file: " << yaml_path << std::endl;
            return {};
        }

        // step2: get image files from config image directory
        std::vector<std::string> image_files = camcalib::ConfigReader::getImageFiles(config.image_dir);
        if (image_files.empty()) {
            std::cerr << "No image files found in directory: " << config.image_dir << std::endl;
            return {};
        }

        std::cout << "Found " << image_files.size() << " image files:" << std::endl;
        for (const auto& file : image_files) {
            std::cout << "  " << file << std::endl;
        }

        std::vector<cv::Mat> gray_images = camcalib::loadGrayImage(image_files);

        return gray_images;
    }
}