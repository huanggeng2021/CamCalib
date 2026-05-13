#include "imageProcess/ImageProcess.h"
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

    std::vector<std::vector<Circle>> runDetectEdge(const std::vector<cv::Mat>& images){

        std::vector<std::vector<Circle>> allCircle;

        std::vector<cv::Mat> edgeImage = detecTheCircleEdge(images);

        std::vector<std::vector<std::vector<cv::Point2i>>> allEdgePoints;
        allEdgePoints = getAllEdgePoints(edgeImage);

        for (int i = 0; i < allEdgePoints.size(); i++){
            std::vector<Circle> oneImageCenter;
            for(int m = 0; m < allEdgePoints[0].size(); m++){

                std::vector<cv::Point2i> oneEdge = allEdgePoints[i][m];

            }

        }

        return allCircle;

    }


    std::vector<cv::Mat> detecTheCircleEdge(const std::vector<cv::Mat>& images){

        int imageNumbers = images.size();

        std::vector<cv::Mat> edgeImages;
        edgeImages.reserve(imageNumbers);

        camcalib::CannyDetecter canny(5, 1.0, 50, 100);

        for(const auto& image:images){

            cv::Mat edges = canny.detect(image);

            edgeImages.push_back(edges);
        }

        return edgeImages;
    }

    std::vector<std::vector<std::vector<cv::Point2i>>> getAllEdgePoints(const std::vector<cv::Mat>& images){

        std::vector<std::vector<std::vector<cv::Point2i>>> allEdgePoints;

        for(const cv::Mat& img:images){

            std::vector<std::vector<cv::Point2i>> edgePoints = getEdgePoints(img);
            allEdgePoints.push_back(edgePoints);
        }

        return allEdgePoints;
    }

    std::vector<std::vector<cv::Point2i>> getEdgePoints(const cv::Mat& edgeImage){

        std::vector<std::vector<cv::Point2i>> edgePoints;
        // Implementation for extracting edge points

        cv::findContours(edgeImage, edgePoints, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // 添加筛选
        return edgePoints;  
    }


    Circle getCircleCenter(const std::vector<cv::Point2i>& points){


    }



}




