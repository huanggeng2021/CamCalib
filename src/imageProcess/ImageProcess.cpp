#include "imageProcess/ImageProcess.h"
#include "utils/Config.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cmath>

namespace camcalib {    

    std::vector<cv::Mat> ImageProcess::loadGrayImage(const std::vector<std::string>& image_paths){

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



    std::vector<cv::Mat> ImageProcess::loadImages(){

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

        std::vector<cv::Mat> gray_images = loadGrayImage(image_files);

        return gray_images;
    }



    std::vector<std::vector<cv::Point>> ImageProcess::filterContours(const std::vector<std::vector<cv::Point>>& contours){

        std::vector<std::vector<cv::Point>> pointCountFilteredContours;

        // 通过边缘点个数筛选
        for(const auto& contour : contours){
            if(contour.size() > 20){
                pointCountFilteredContours.push_back(contour);
            }
        }

        std::vector<std::vector<cv::Point>> circleFilteredContours;

        // 通过圆的圆度筛选 即长轴与短轴的比例
        for(const auto& contour : pointCountFilteredContours){
            if(contour.empty()){
                continue;
            }

            int minx = contour[0].x;
            int maxx = contour[0].x;
            int miny = contour[0].y;
            int maxy = contour[0].y;

            for(const auto& point : contour){
                minx = std::min(minx, point.x);
                maxx = std::max(maxx, point.x);
                miny = std::min(miny, point.y);
                maxy = std::max(maxy, point.y);
            }

            int width = maxx - minx + 1;
            int height = maxy - miny + 1;
            int longAxis = std::max(width, height);
            int shortAxis = std::min(width, height);

            double axisRatio = static_cast<double>(longAxis) / static_cast<double>(shortAxis);

            if(axisRatio <= 1.5){
                circleFilteredContours.push_back(contour);
            }
        }

        return circleFilteredContours;
    }



    std::vector<std::vector<std::vector<cv::Point>>> ImageProcess::runDetectEdge(const std::vector<cv::Mat>& images){

        // 使用阈值分割的方式提取边缘点

        std::vector<std::vector<std::vector<cv::Point>>> allCountours;
        allCountours.reserve(images.size());

        for(int i = 0; i< images.size(); i++){

            cv::Mat binary;
            cv::threshold(images[i], binary, 0, 255, cv::THRESH_BINARY_INV  | cv::THRESH_OTSU);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            std::vector<std::vector<cv::Point>> filterdContours = filterContours(contours);

            std::cout << filterdContours.size() << std::endl;
            allCountours.push_back(filterdContours);
        }
        

        return allCountours;
    }



    void ImageProcess::showEdgeAndCircleCenters(const cv::Mat& binary, const std::vector<std::vector<cv::Point>>& contours, const std::vector<Circle>& centerPoints, const std::string& windowName){

        cv::Mat edgeDisplay;
        cv::cvtColor(binary, edgeDisplay, cv::COLOR_GRAY2BGR);
        cv::drawContours(edgeDisplay, contours, -1, cv::Scalar(0, 0, 255), 3);
        drawCenterPoints(edgeDisplay, centerPoints);

        for(size_t j = 0; j < contours.size(); j++){
            cv::Rect rect = cv::boundingRect(contours[j]);
            cv::Point textPos(rect.x, std::max(15, rect.y - 5));

            cv::putText(
                edgeDisplay,
                std::to_string(j),
                textPos,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 255, 0),
                1,
                cv::LINE_AA
            );
        }

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 1000);
        cv::imshow(windowName, edgeDisplay);
        cv::waitKey(0);

    }


    void ImageProcess::drawCenterPoints(cv::Mat& image, const std::vector<Circle>& centerPoints){

        for(const auto& circle : centerPoints){
            if(circle.radius <= 0.0){
                continue;
            }

            cv::Point center(static_cast<int>(std::round(circle.center.x)), static_cast<int>(std::round(circle.center.y)));
            cv::circle(image, center, 4, cv::Scalar(255, 0, 0), -1);
            cv::drawMarker(image, center, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 16, 2);
        }

    }


    ImageProcess::Circle ImageProcess::getCircleCenter(const std::vector<cv::Point2i>& points){

        int N = points.size();
        if(N < 3){
            Circle c;
            c.edge_points = points;
            c.center = cv::Point2d(0.0, 0.0);
            c.radius = 0.0;
            return c;
        }

        Eigen::MatrixXd A(N,3);
        Eigen::MatrixXd B(N,1);

        for(int i = 0; i < N; i++){
            double x = static_cast<double>(points[i].x);
            double y = static_cast<double>(points[i].y);
            A(i, 0) = x;
            A(i, 1) = y;
            A(i, 2) = 1.0;

            B(i,0) =  static_cast<double>(x * x + y * y);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd S = svd.singularValues();

        Eigen::MatrixXd Sigma_pinv = Eigen::MatrixXd::Zero(V.cols(), U.rows());

        double tolerance = 1e-10;

        for(int i = 0; i < S.size(); i++){
            if (S(i) > tolerance){
                Sigma_pinv(i,i) = 1.0 / S(i);
            }
        }

        Eigen::MatrixXd X = V * Sigma_pinv * U.transpose();
        Eigen::VectorXd params = X * B;

        double D = params(0);
        double E = params(1);
        double F = params(2);

        Circle c;
        c.edge_points = points;
        c.center = cv::Point2d(D / 2.0, E / 2.0);
        double radius_sq = c.center.x * c.center.x + c.center.y * c.center.y + F;
        c.radius = radius_sq > 0.0 ? std::sqrt(radius_sq) : 0.0;

        return c;

    }


    // 
    void ImageProcess::runCalibrate(){

        // 加载灰度图像
        std::vector<cv::Mat> images = loadImages();

        // 图像预处理
        std::vector<std::vector<std::vector<cv::Point>>> coners = runDetectEdge(images);

        // 对圆心进行排序

        // 检测边缘
        for(int i = 0; i < coners.size(); i++){

            std::vector<Circle> centerPoints;
            centerPoints.reserve(coners[i].size());

            for(int j = 0; j < coners[i].size(); j++){

                Circle c = getCircleCenter(coners.at(i).at(j));
                centerPoints.push_back(c);

            }

            if(i == 6){
                cv::Mat binary;
                cv::threshold(images[i], binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
                showEdgeAndCircleCenters(binary, coners[i], centerPoints, "Detected Edges " + std::to_string(i));
            }

        }


        // 拟合圆心

        // 计算坐标系下的世界坐标点

        // 计算标定参数

        // 计算重投影误差 评估质量


    }

}
