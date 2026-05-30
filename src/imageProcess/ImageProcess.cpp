#include "imageProcess/ImageProcess.h"
#include "utils/Config.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <limits>

namespace camcalib {    

    std::vector<ImageProcess::Circle> ImageProcess::getBigMarkers(const std::vector<ImageProcess::Circle>& circles){

        std::vector<ImageProcess::Circle> sortedCircles = circles;

        std::sort(sortedCircles.begin(), sortedCircles.end(), []
        (const ImageProcess::Circle& a, const ImageProcess::Circle& b){
             return a.radius > b.radius;
            }
        );

        std::vector<ImageProcess::Circle> bigMarkers;
        for (int k = 0; k < 5 && k < sortedCircles.size(); k++)
        {
            bigMarkers.push_back(sortedCircles[k]);
        }

        return bigMarkers;
    }

    ImageProcess::MarkerDistanceInfo ImageProcess::findNearestAndFarthestMarkers(const std::vector<ImageProcess::Circle>& bigMarkers){

        MarkerDistanceInfo distanceInfo;
        double minDistance = std::numeric_limits<double>::max(); 
        double maxDistance = std::numeric_limits<double>::lowest();

        for(int mi = 0; mi < 5; mi++ ){
            for(int mj = mi + 1; mj < 5; mj++){

                double x = static_cast<double>(bigMarkers[mi].center.x - bigMarkers[mj].center.x);
                double y = static_cast<double>(bigMarkers[mi].center.y - bigMarkers[mj].center.y);
                double dis = sqrt(x * x + y * y);
                
                if(dis < minDistance){

                    minDistance = dis;
                    distanceInfo.nearest.first = mi;
                    distanceInfo.nearest.second = mj;

                }
                
                if(dis > maxDistance){

                    maxDistance = dis;
                    distanceInfo.farthest.first = mi;
                    distanceInfo.farthest.second = mj;

                }
 
            }

        }

        return distanceInfo;
    }

    int ImageProcess::findRemainingMarkerIndex(const MarkerDistanceInfo& distanceInfo){

        for(int pi = 0; pi < 5; pi++){
            if( pi != distanceInfo.nearest.first &&
                pi != distanceInfo.nearest.second &&
                pi != distanceInfo.farthest.first &&
                pi != distanceInfo.farthest.second){

                return pi;

             }
        }

        return -1;
    }

    bool ImageProcess::isMarkerPairParallel(
        const std::vector<ImageProcess::Circle>& bigMarkers,
        const MarkerDistanceInfo& distanceInfo
    ){

        cv::Point2d nearP1 = bigMarkers[distanceInfo.nearest.first].center;
        cv::Point2d nearP2 = bigMarkers[distanceInfo.nearest.second].center;
        cv::Point2d farP1 = bigMarkers[distanceInfo.farthest.first].center;
        cv::Point2d farP2 = bigMarkers[distanceInfo.farthest.second].center;

        cv::Point2d v1 = nearP2 - nearP1;
        cv::Point2d v2 = farP2 - farP1;

        Eigen::Vector2d v11(v1.x, v1.y);
        Eigen::Vector2d v22(v2.x, v2.y);

        double dot = v11.dot(v22);
        double cross = v11.x() * v22.y() - v11.y() * v22.x();

        double angle = std::atan2(cross, dot) * 180.0 / M_PI;
        double parallelAngle = std::abs(angle);
        parallelAngle = 180.0 - parallelAngle;
        
        if(parallelAngle > 90){
            parallelAngle = 180.0 - parallelAngle;
        }

        std::cout << "angle = " << parallelAngle << std::endl;

        return parallelAngle <= 10.0;
    }

    std::vector<ImageProcess::Circle> ImageProcess::orderBigMarkers(
        const std::vector<ImageProcess::Circle>& bigMarkers,
        const MarkerDistanceInfo& distanceInfo,
        int p3Index
    ){

        int near1 = distanceInfo.nearest.first;
        int near2 = distanceInfo.nearest.second;

        int far1 = distanceInfo.farthest.first;
        int far2 = distanceInfo.farthest.second;

        cv::Point2d nearP1 = bigMarkers[near1].center;
        cv::Point2d nearP2 = bigMarkers[near2].center;

        cv::Point2d farP1 = bigMarkers[far1].center;
        cv::Point2d farP2 = bigMarkers[far2].center;

        cv::Point2d p3 = bigMarkers[p3Index].center;

        cv::Point2d farVector = farP2 - farP1;
        Eigen::Vector2d v22(farVector.x, farVector.y);

        Eigen::Vector2d vNear1P3(p3.x - nearP1.x, p3.y - nearP1.y);
        Eigen::Vector2d vNear2P3(p3.x - nearP2.x, p3.y - nearP2.y);

        double angle1 = std::atan2(vNear1P3.x() * v22.y() - vNear1P3.y() * v22.x(), 
                                vNear1P3.dot(v22)) * 180.0 / CV_PI;

        double angle2 = std::atan2(vNear2P3.x() * v22.y() - vNear2P3.y() * v22.x(),
                                vNear2P3.dot(v22)) * 180.0 / CV_PI;

        int p00 = 0;
        int p11 = 0;
        int p22 = 0;
        int p33 = p3Index;
        int p44 = 0;              

        if(angle1 > angle2){
            p00 = near1;
            p11 = near2;
        }
        if(angle1 < angle2){
            p00 = near2;
            p11 = near1;
        } 
        
        double distance1 = std::pow(bigMarkers[p00].center.x - farP1.x, 2) + 
                            std::pow(bigMarkers[p00].center.y - farP1.y, 2);

        double distance2 = std::pow(bigMarkers[p00].center.x - farP2.x, 2) + 
                            std::pow(bigMarkers[p00].center.y - farP2.y, 2);

        if(distance1 > distance2){
            p22 = far1;
            p44 = far2;
        }

        if(distance1 < distance2){
            p22 = far2;
            p44 = far1;
        }

        std::vector<ImageProcess::Circle> orderedMarkers;

        orderedMarkers.push_back(bigMarkers[p00]);
        orderedMarkers.push_back(bigMarkers[p11]);
        orderedMarkers.push_back(bigMarkers[p22]);
        orderedMarkers.push_back(bigMarkers[p33]);
        orderedMarkers.push_back(bigMarkers[p44]);

        return orderedMarkers;
    }

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

    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::sortCircleCenter(std::vector<std::vector<Circle>>& disorderedCenter){

        std::vector<std::vector<ImageProcess::Circle>> sortedAllImages;
        int numSize = static_cast<int>(disorderedCenter.size());

        for(int i = 0; i < numSize; i++){

            std::vector<ImageProcess::Circle> bigMarkers = getBigMarkers(disorderedCenter[i]);

            if(bigMarkers.size() < 5){
                std::cerr << "第 " << i << " 张图标志点数量不足 5 个" << std::endl;
                sortedAllImages.push_back({});
                continue;
            }

            MarkerDistanceInfo distanceInfo = findNearestAndFarthestMarkers(bigMarkers);
            int p3Index = findRemainingMarkerIndex(distanceInfo);

            if (p3Index == -1)
            {
                std::cerr << "p3 查找失败，大圆索引关系异常" << std::endl;
                
                sortedAllImages.push_back({});
                continue;
                
            }

            if (!isMarkerPairParallel(bigMarkers, distanceInfo))
            {
                std::cerr << "第 " << i << " 张图标志点寻找错误：最近点对与最远点对不平行" << std::endl;
                sortedAllImages.push_back({});
                continue;
            }

            sortedAllImages.push_back(orderBigMarkers(bigMarkers, distanceInfo, p3Index));

        }


        return sortedAllImages;

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


    void ImageProcess::showSortedCircleCenters(const cv::Mat& image, const std::vector<Circle>& sortedCenterPoints, const std::string& windowName){

        cv::Mat display;
        if(image.channels() == 1){
            cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
        }else{
            display = image.clone();
        }

        for(size_t i = 0; i < sortedCenterPoints.size(); i++){
            const Circle& circle = sortedCenterPoints[i];
            if(circle.radius <= 0.0){
                continue;
            }

            cv::Point center(
                static_cast<int>(std::round(circle.center.x)),
                static_cast<int>(std::round(circle.center.y))
            );

            cv::circle(display, center, static_cast<int>(std::round(circle.radius)), cv::Scalar(0, 255, 255), 2);
            cv::circle(display, center, 4, cv::Scalar(0, 0, 255), -1);
            cv::drawMarker(display, center, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 16, 2);

            std::string text = std::to_string(i) + " (" +
                               std::to_string(static_cast<int>(std::round(circle.center.x))) + ", " +
                               std::to_string(static_cast<int>(std::round(circle.center.y))) + ")";

            cv::Point textPos(center.x + 8, center.y - 8);
            textPos.x = std::max(0, std::min(textPos.x, display.cols - 160));
            textPos.y = std::max(20, std::min(textPos.y, display.rows - 5));

            cv::putText(
                display,
                text,
                textPos,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA
            );
        }

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 1000);
        cv::imshow(windowName, display);
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

        // 图像预处理 检测边缘
        std::vector<std::vector<std::vector<cv::Point>>> coners = runDetectEdge(images);

        // 拟合圆心
        std::vector<std::vector<Circle>> allCenterPoints;

        for(int i = 0; i < coners.size(); i++){

            std::vector<Circle> centerPoints;
            centerPoints.reserve(coners[i].size());

            for(int j = 0; j < coners[i].size(); j++){

                Circle c = getCircleCenter(coners.at(i).at(j));
                centerPoints.push_back(c);

            }
            allCenterPoints.push_back(centerPoints);
            //cv::Mat binary;
            //cv::threshold(images[i], binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            //showEdgeAndCircleCenters(binary, coners[i], centerPoints, "Detected Edges " + std::to_string(i));
        }

        // 对圆心进行排序 
        std::vector<std::vector<Circle>> sortCenterPoints = sortCircleCenter(allCenterPoints);

        for(size_t i = 0; i < sortCenterPoints.size() && i < images.size(); i++){
            if(sortCenterPoints[i].empty()){
                continue;
            }

            showSortedCircleCenters(
                images[i],
                sortCenterPoints[i],
                "Sorted Circle Centers " + std::to_string(i)
            );
        }

        // 拟合圆心

        // 计算坐标系下的世界坐标点

        // 计算标定参数

        // 计算重投影误差 评估质量


    }

}
