#include "imageProcess/ImageProcess.h"
#include "utils/Config.h"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

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

        // std::cout << "angle = " << parallelAngle << std::endl;

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
                logError("Failed to load image: " + path);
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

    cv::Point2d ImageProcess::applyHomography(const Eigen::Matrix3d& homography, const cv::Point2d& point){
        Eigen::Vector3d p(point.x, point.y, 1.0);
        Eigen::Vector3d projected = homography * p;

        if(std::abs(projected(2)) < 1e-12){
            return cv::Point2d(0.0, 0.0);
        }

        return cv::Point2d(
            projected(0) / projected(2),
            projected(1) / projected(2)
        );
    }

    Eigen::Matrix3d ImageProcess::cvMatToEigenMat3d(const cv::Mat& mat){
        Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
        for(int row = 0; row < 3; ++row){
            for(int col = 0; col < 3; ++col){
                result(row, col) = mat.at<double>(row, col);
            }
        }
        return result;
    }

    void ImageProcess::printHomographyValidation(
        int imageIndex,
        const std::vector<cv::Point2d>& idealCenter,
        const std::vector<Circle>& sortedCircleCenter,
        const Eigen::Matrix3d& customH,
        const Eigen::Matrix3d& cvH
    ){
        std::cout << "Image " << imageIndex << " control point reprojection:" << std::endl;
        for(size_t i = 0; i < idealCenter.size() && i < sortedCircleCenter.size(); ++i){
            const cv::Point2d& idealPoint = idealCenter[i];
            const cv::Point2d& imagePoint = sortedCircleCenter[i].center;
            cv::Point2d customProjected = applyHomography(customH, idealPoint);
            cv::Point2d cvProjected = applyHomography(cvH, idealPoint);

            double customError = cv::norm(customProjected - imagePoint);
            double cvError = cv::norm(cvProjected - imagePoint);

            std::cout << "  [" << i << "] ideal(" << idealPoint.x << ", " << idealPoint.y << ")"
                      << " -> image measured(" << imagePoint.x << ", " << imagePoint.y << ")"
                      << " custom(" << customProjected.x << ", " << customProjected.y << ")"
                      << " err=" << customError
                      << " cv(" << cvProjected.x << ", " << cvProjected.y << ")"
                      << " err=" << cvError
                      << std::endl;
        }
    }

    void ImageProcess::printCircleList(const std::string& title, const std::vector<Circle>& circles){
        std::cout << title << " size = " << circles.size() << std::endl;
        for(size_t i = 0; i < circles.size(); ++i){
            const auto& circle = circles[i];
            std::cout << "  [" << i << "]"
                      << " image(x=" << circle.center.x
                      << ", y=" << circle.center.y << ")"
                      << " radius=" << circle.radius
                      << " ideal(row=" << circle.homo_row
                      << ", col=" << circle.homo_col << ")"
                      << std::endl;
        }
    }



    std::vector<cv::Mat> ImageProcess::loadImages(const CaliConfig& config){
        // step2: get image files from config image directory
        std::vector<std::string> image_files = camcalib::ConfigReader::getImageFiles(
            config.image_dir,
            config.image_extensions
        );
        if (image_files.empty()) {
            logError("No image files found in directory: " + config.image_dir);
            return {};
        }

        logInfo("Found " + std::to_string(image_files.size()) + " image files in " + config.image_dir);
        for (const auto& file : image_files) {
            logInfo("Image file: " + file);
        }

        if(config.read_grayscale){
            return loadGrayImage(image_files);
        }

        std::vector<cv::Mat> images;
        images.reserve(image_files.size());

        for(const auto& path : image_files){
            cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
            if(image.empty()){
                logError("Failed to load image: " + path);
                continue;
            }
            images.push_back(image);
        }

        return images;
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

            cv::Mat binary = buildBinaryImage(images[i]);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            std::vector<std::vector<cv::Point>> filterdContours = filterContours(contours);

            allCountours.push_back(filterdContours);
        }
        

        return allCountours;
    }

    std::vector<std::vector<std::vector<cv::Point2f>>> ImageProcess::runDetectSubPixelEdge(const std::vector<cv::Mat>& images){

        // 检测边缘

        // 亚像素细化

        return {};

    }



    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::sortCircleCenter(std::vector<std::vector<Circle>>& disorderedCenter){

        std::vector<std::vector<ImageProcess::Circle>> sortedAllImages;
        int numSize = static_cast<int>(disorderedCenter.size());

        for(int i = 0; i < numSize; i++){

            std::vector<ImageProcess::Circle> bigMarkers = getBigMarkers(disorderedCenter[i]);
            //printCircleList("Image " + std::to_string(i) + " big markers(before order)", bigMarkers);

            if(bigMarkers.size() < 5){
                logError("Image " + std::to_string(i) + " does not have enough marker circles.");
                sortedAllImages.push_back({});
                continue;
            }

            MarkerDistanceInfo distanceInfo = findNearestAndFarthestMarkers(bigMarkers);
            int p3Index = findRemainingMarkerIndex(distanceInfo);

            if (p3Index == -1)
            {
                logError("Image " + std::to_string(i) + " failed to find p3 marker.");
                
                sortedAllImages.push_back({});
                continue;
                
            }

            if (!isMarkerPairParallel(bigMarkers, distanceInfo))
            {
                logError("Image " + std::to_string(i) + " marker ordering failed because marker pairs are not parallel.");
                sortedAllImages.push_back({});
                continue;
            }

            std::vector<ImageProcess::Circle> orderedMarkers = orderBigMarkers(bigMarkers, distanceInfo, p3Index);
            //printCircleList("Image " + std::to_string(i) + " ordered big markers", orderedMarkers);
            sortedAllImages.push_back(orderedMarkers);

        }


        return sortedAllImages;

    }


    std::vector<Eigen::Matrix3d> ImageProcess::findHomography(const std::vector<std::vector<ImageProcess::Circle>>& sortedCircleCenter){

        int iNum = sortedCircleCenter.size();

        std::vector<Eigen::Matrix3d> homographyVec;
        homographyVec.reserve(iNum);

        // 定义标准坐标系位置
        double dCenterDistance = 150.0;

        cv::Point2d p0(1 * dCenterDistance, 2 * dCenterDistance);
        cv::Point2d p1(0 * dCenterDistance, 2 * dCenterDistance);
        cv::Point2d p2(-3 * dCenterDistance, 0 * dCenterDistance);
        cv::Point2d p3(0 * dCenterDistance, -2 * dCenterDistance);
        cv::Point2d p4(3 * dCenterDistance, 0 * dCenterDistance);

        std::vector<cv::Point2d> idealCenter;
        idealCenter.push_back(p0);
        idealCenter.push_back(p1);
        idealCenter.push_back(p2);
        idealCenter.push_back(p3);
        idealCenter.push_back(p4);

       

        for(int iImgNum = 0; iImgNum < iNum; iImgNum++){

            Eigen::MatrixXd A(2 * 5, 9);
            A.setZero();
            std::vector<cv::Point2d> imageCenters;
            imageCenters.reserve(5);

            for (int iCircle = 0; iCircle < 5; iCircle++){

                double X = idealCenter[iCircle].x;
                double Y = idealCenter[iCircle].y;
                
                double u = sortedCircleCenter[iImgNum][iCircle].center.x;
                double v = sortedCircleCenter[iImgNum][iCircle].center.y;
                imageCenters.push_back(sortedCircleCenter[iImgNum][iCircle].center);

                A(iCircle * 2, 0) = X;
                A(iCircle * 2, 1) = Y;
                A(iCircle * 2, 2) = 1;
                A(iCircle * 2, 3) = 0;
                A(iCircle * 2, 4) = 0;
                A(iCircle * 2, 5) = 0;
                A(iCircle * 2, 6) = -u * X;
                A(iCircle * 2, 7) = -u * Y;
                A(iCircle * 2, 8) = -u;

                A(iCircle * 2 + 1, 0) = 0;
                A(iCircle * 2 + 1, 1) = 0;
                A(iCircle * 2 + 1, 2) = 0;
                A(iCircle * 2 + 1, 3) = X;
                A(iCircle * 2 + 1, 4) = Y;
                A(iCircle * 2 + 1, 5) = 1;
                A(iCircle * 2 + 1, 6) = -v * X;
                A(iCircle * 2 + 1, 7) = -v * Y;
                A(iCircle * 2 + 1, 8) = -v;


            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

            Eigen::VectorXd h = svd.matrixV().col(8);

            Eigen::Matrix3d H;
            H << h(0), h(1), h(2),
                h(3), h(4), h(5),
                h(6), h(7), h(8);

            // 归一化，让 H(2,2) = 1
            if (std::abs(H(2, 2)) > 1e-12)
            {
                    H = H / H(2, 2);
            }

            cv::Mat cvH = cv::findHomography(idealCenter, imageCenters, 0);
            Eigen::Matrix3d cvHomography = cvMatToEigenMat3d(cvH);

            // std::cout << "Image " << iImgNum << " homography H:" << std::endl;   调试信息
            // std::cout << H << std::endl;
            // std::cout << "Image " << iImgNum << " OpenCV homography Hcv:" << std::endl;
            // std::cout << cvHomography << std::endl;
            // printHomographyValidation(iImgNum, idealCenter, sortedCircleCenter[iImgNum], H, cvHomography); 
            homographyVec.push_back(H);


        }

        return homographyVec;

    }


    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::homographyCircleCenter(
        const std::vector<Eigen::Matrix3d>& homo,
        const std::vector<std::vector<ImageProcess::Circle>>& unSortCircleCenter
    ){

        int iNum = unSortCircleCenter.size();

        std::vector<std::vector<ImageProcess::Circle>> sortCircleCenter;
        sortCircleCenter.reserve(iNum);

    
        if(unSortCircleCenter.size() != homo.size()){
            return {};
        }

        for(int iImg = 0; iImg < iNum; iImg++){

            Eigen::MatrixXd h_inv = homo[iImg].inverse();
            std::vector<Circle> homoCenterVec;
            homoCenterVec.clear();

            for(int iCircle = 0; iCircle < unSortCircleCenter[iImg].size(); iCircle++){

                cv::Point2d imagePoint =  unSortCircleCenter[iImg][iCircle].center;
                Circle c = unSortCircleCenter[iImg][iCircle];

                Eigen::Vector3d p_img;
                p_img << imagePoint.x, imagePoint.y, 1.0;

                Eigen::Vector3d p_ideal_h = h_inv * p_img;

                if (std::abs(p_ideal_h(2)) < 1e-12)
                {
                    throw std::runtime_error("Homogeneous coordinate is too close to zero.");
                }

                double X = p_ideal_h(0) / p_ideal_h(2);
                double Y = p_ideal_h(1) / p_ideal_h(2);

                // OpenCV 点坐标使用 (x, y) = (col, row)，排序时要按 row 再按 col。
                c.homo_col = X;
                c.homo_row = Y;
                
                homoCenterVec.push_back(c);
            }

            // printCircleList("Image " + std::to_string(iImg) + " all centers(before homography sort)", homoCenterVec);

            // 排序
            // 先按行 homo_row 从小到大，再按列 homo_col 从小到大
            double dCenterDistance = 15.0;
            double rowTolerance = dCenterDistance * 0.5;

            std::sort(homoCenterVec.begin(), homoCenterVec.end(),
                [rowTolerance](const Circle& a, const Circle& b)
                {
                    if (std::abs(a.homo_row - b.homo_row) > rowTolerance)
                    {
                        return a.homo_row < b.homo_row;
                    }

                    return a.homo_col < b.homo_col;
                });

            // printCircleList("Image " + std::to_string(iImg) + " all centers(after homography sort)", homoCenterVec);
            sortCircleCenter.push_back(homoCenterVec);
        }

        return sortCircleCenter;

    };

    
    std::vector<std::vector<cv::Point3f>> ImageProcess::generateWorldCoordinates(
        const int imageNum,const int calibRows,const int calibCols, const double centerDist){

        std::vector<std::vector<cv::Point3f>> worldPointsVec;
        worldPointsVec.reserve(imageNum);

        std::vector<cv::Point3f> singleWorldPoints;
        singleWorldPoints.reserve(calibRows * calibCols);

        // 中心偏置
        double centerOffsetX = (calibCols / 2) * centerDist;  
        double centerOffsetY = (calibRows / 2) * centerDist;

        for(int rows = 0; rows < calibRows; rows++){
            for(int cols = 0; cols < calibCols; cols++){

                double x = cols * centerDist - centerOffsetX;  // 列坐标
                double y = rows * centerDist - centerOffsetY;  // 行坐标
                double z = 0;
                
                cv::Point3d p(x, y, z);

                singleWorldPoints.push_back(p);
            }
        }

        worldPointsVec.resize(imageNum, singleWorldPoints);

        return worldPointsVec;
    }

    cv::Mat ImageProcess::buildBinaryImage(const cv::Mat& image){

        cv::Mat gray;
        if(image.channels() == 1){
            gray = image;
        }else{
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }

        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        return binary;
    }

    cv::Mat ImageProcess::renderEdgeAndCircleCenters(
        const cv::Mat& binary,
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<Circle>& centerPoints
    ){

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

        return edgeDisplay;
    }

    cv::Mat ImageProcess::renderSortedCircleCenters(
        const cv::Mat& image,
        const std::vector<Circle>& sortedCenterPoints
    ){

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

            std::string text = std::to_string(i);

            cv::Point textPos(center.x + 8, center.y - 8);
            textPos.x = std::max(0, std::min(textPos.x, display.cols - 40));
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

        return display;
    }

    void ImageProcess::initializeLogger(const CaliConfig& config){

        log_enabled_ = config.log_enabled;
        if(!log_enabled_){
            return;
        }

        std::filesystem::path logPath = config.log_output_file;
        try{
            if(logPath.has_parent_path()){
                std::filesystem::create_directories(logPath.parent_path());
            }
            log_stream_.open(logPath, std::ios::out | std::ios::app);
        }catch(const std::filesystem::filesystem_error& e){
            std::cerr << "Failed to prepare log file: " << e.what() << std::endl;
            log_enabled_ = false;
            return;
        }

        if(!log_stream_.is_open()){
            std::cerr << "Failed to open log file: " << logPath << std::endl;
            log_enabled_ = false;
        }
    }

    void ImageProcess::shutdownLogger(){

        if(log_stream_.is_open()){
            log_stream_.flush();
            log_stream_.close();
        }
        log_enabled_ = false;
    }

    void ImageProcess::logMessage(const std::string& level, const std::string& message){

        std::time_t now = std::time(nullptr);
        std::tm* localTime = std::localtime(&now);

        std::ostringstream stream;
        if(localTime != nullptr){
            stream << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
        }else{
            stream << "unknown-time";
        }
        stream << " [" << level << "] " << message;

        if(level == "ERROR"){
            std::cerr << stream.str() << std::endl;
        }else{
            std::cout << stream.str() << std::endl;
        }

        if(log_enabled_ && log_stream_.is_open()){
            log_stream_ << stream.str() << std::endl;
        }
    }

    void ImageProcess::logInfo(const std::string& message){
        logMessage("INFO", message);
    }

    void ImageProcess::logError(const std::string& message){
        logMessage("ERROR", message);
    }

    bool ImageProcess::saveDebugImage(const std::filesystem::path& outputPath, const cv::Mat& image){

        try{
            std::filesystem::create_directories(outputPath.parent_path());
        }catch(const std::filesystem::filesystem_error& e){
            logError("Failed to create debug directory: " + std::string(e.what()));
            return false;
        }

        if(!cv::imwrite(outputPath.string(), image)){
            logError("Failed to save debug image: " + outputPath.string());
            return false;
        }

        return true;
    }


    void ImageProcess::showEdgeAndCircleCenters(const cv::Mat& binary, const std::vector<std::vector<cv::Point>>& contours, const std::vector<Circle>& centerPoints, const std::string& windowName){

        cv::Mat edgeDisplay = renderEdgeAndCircleCenters(binary, contours, centerPoints);

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 1000);
        cv::imshow(windowName, edgeDisplay);
        cv::waitKey(0);

    }


    void ImageProcess::showSortedCircleCenters(const cv::Mat& image, const std::vector<Circle>& sortedCenterPoints, const std::string& windowName){

        cv::Mat display = renderSortedCircleCenters(image, sortedCenterPoints);

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 1000);
        cv::imshow(windowName, display);
        cv::waitKey(0);

    }

    void ImageProcess::showWarpedImage(const cv::Mat& image, const Eigen::Matrix3d& homography, const std::string& windowName){

        cv::Mat display;
        if(image.channels() == 1){
            cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
        }else{
            display = image.clone();
        }

        cv::Mat hCv(3, 3, CV_64F);
        for(int row = 0; row < 3; ++row){
            for(int col = 0; col < 3; ++col){
                hCv.at<double>(row, col) = homography(row, col);
            }
        }

        cv::Mat hInv = hCv.inv();

        std::vector<cv::Point2f> srcCorners = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(static_cast<float>(display.cols - 1), 0.0f),
            cv::Point2f(static_cast<float>(display.cols - 1), static_cast<float>(display.rows - 1)),
            cv::Point2f(0.0f, static_cast<float>(display.rows - 1))
        };

        std::vector<cv::Point2f> warpedCorners;
        cv::perspectiveTransform(srcCorners, warpedCorners, hInv);

        float minX = warpedCorners[0].x;
        float maxX = warpedCorners[0].x;
        float minY = warpedCorners[0].y;
        float maxY = warpedCorners[0].y;
        for(const auto& point : warpedCorners){
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
        }

        cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
        translation.at<double>(0, 2) = -minX;
        translation.at<double>(1, 2) = -minY;

        int warpedWidth = std::max(1, static_cast<int>(std::ceil(maxX - minX)));
        int warpedHeight = std::max(1, static_cast<int>(std::ceil(maxY - minY)));

        cv::Mat warped;
        cv::warpPerspective(
            display,
            warped,
            translation * hInv,
            cv::Size(warpedWidth, warpedHeight),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1200, 1000);
        cv::imshow(windowName, warped);
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
        c.center = cv::Point2d(D / 2.0, E / 2.0);   // 圆心 xy  与cv::point2d中的 xy并不一致
        double radius_sq = c.center.x * c.center.x + c.center.y * c.center.y + F;
        c.radius = radius_sq > 0.0 ? std::sqrt(radius_sq) : 0.0;

        return c;

    }


    // 
    void ImageProcess::runCalibrate(){

        std::string yaml_path = "config/calib_config.yaml";

        CaliConfig config;
        if (!ConfigReader::readConfig(yaml_path, config)) {
            std::cerr << "Failed to read config file: " << yaml_path << std::endl;
            return;
        }

        initializeLogger(config);
        logInfo("Calibration started.");
        logInfo("Current working directory: " + std::filesystem::current_path().string());
        logInfo("Loaded config file: " + yaml_path);

        const bool shouldSaveDebugImages = config.debug_mode && config.debug_save_images;
        const bool shouldShowDebugWindows = config.debug_mode && config.debug_show_windows;
        logInfo(
            "Options: log_enabled=" + std::to_string(config.log_enabled) +
            ", debug_mode=" + std::to_string(config.debug_mode) +
            ", debug_save_images=" + std::to_string(config.debug_save_images) +
            ", debug_show_windows=" + std::to_string(config.debug_show_windows)
        );

        // 加载灰度图像
        std::vector<cv::Mat> images = loadImages(config);
        if(images.empty()){
            logError("Calibration aborted because no input images were loaded.");
            shutdownLogger();
            return;
        }

        std::filesystem::path debugRoot = config.debug_output_dir;
        if(shouldSaveDebugImages){
            try{
                std::filesystem::create_directories(debugRoot);
            }catch(const std::filesystem::filesystem_error& e){
                logError("Failed to create debug output directory: " + std::string(e.what()));
                shutdownLogger();
                return;
            }
            logInfo("Debug images will be saved to: " + debugRoot.string());
        }

        // 图像预处理 检测边缘
        std::vector<std::vector<std::vector<cv::Point>>> coners = runDetectEdge(images);
        logInfo("Edge detection completed for " + std::to_string(coners.size()) + " images.");

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
            logInfo(
                "Image " + std::to_string(i) +
                ": contours=" + std::to_string(coners[i].size()) +
                ", fitted_centers=" + std::to_string(centerPoints.size())
            );

            if(shouldSaveDebugImages){
                std::ostringstream imageFolderName;
                imageFolderName << "image_" << std::setw(3) << std::setfill('0') << i;
                std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

                cv::Mat binary = buildBinaryImage(images[i]);
                saveDebugImage(
                    imageDebugDir / "01_detected_edges.png",
                    renderEdgeAndCircleCenters(binary, coners[i], centerPoints)
                );
                saveDebugImage(
                    imageDebugDir / "02_fitted_centers.png",
                    renderSortedCircleCenters(images[i], centerPoints)
                );
                logInfo("Saved debug images for image " + std::to_string(i) + " fitted centers stage.");
            }
        }

        // 对圆心进行排序 
        std::vector<std::vector<Circle>> sortCenterPoints = sortCircleCenter(allCenterPoints);
        logInfo("Marker sorting completed.");

        if(shouldSaveDebugImages){
            for(size_t i = 0; i < sortCenterPoints.size() && i < images.size(); i++){
                if(sortCenterPoints[i].empty()){
                    logError("Image " + std::to_string(i) + " has no sorted marker output.");
                    continue;
                }

                std::ostringstream imageFolderName;
                imageFolderName << "image_" << std::setw(3) << std::setfill('0') << i;
                std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

                saveDebugImage(
                    imageDebugDir / "03_sorted_markers.png",
                    renderSortedCircleCenters(images[i], sortCenterPoints[i])
                );
                logInfo("Saved sorted marker debug image for image " + std::to_string(i) + ".");
            }
        }

        if(shouldShowDebugWindows){
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
        }

        //计算变换矩阵
        std::vector<Eigen::Matrix3d> homoVec = findHomography(sortCenterPoints);
        logInfo("Homography computation completed.");

        std::vector<std::vector<ImageProcess::Circle>> sortedCircleCenter = homographyCircleCenter(homoVec, allCenterPoints);
        logInfo("Full board sorting completed.");

        if(shouldSaveDebugImages){
            for(size_t i = 0; i < sortedCircleCenter.size() && i < images.size(); i++){
                if(sortedCircleCenter[i].empty()){
                    logError("Image " + std::to_string(i) + " has no final sorted board output.");
                    continue;
                }

                std::ostringstream imageFolderName;
                imageFolderName << "image_" << std::setw(3) << std::setfill('0') << i;
                std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

                saveDebugImage(
                    imageDebugDir / "04_sorted_board.png",
                    renderSortedCircleCenters(images[i], sortedCircleCenter[i])
                );
                logInfo("Saved final sorted board debug image for image " + std::to_string(i) + ".");
            }
        }

        if(shouldShowDebugWindows){
            for(size_t i = 0; i < sortedCircleCenter.size() && i < images.size(); i++){
                if(sortedCircleCenter[i].empty()){
                    continue;
                }

                showWarpedImage(
                    images[i],
                    homoVec[i],
                    "Warped Image " + std::to_string(i)
                );

                showSortedCircleCenters(
                    images[i],
                    sortedCircleCenter[i],
                    "Sorted Circle Centers " + std::to_string(i)
                );
            }
        }

        // 计算坐标系下的世界坐标点
        std::vector<std::vector<cv::Point3f>> worldCorrdianates = generateWorldCoordinates(
            sortedCircleCenter.size(),
            config.calib_rows,
            config.calib_cols,
            config.calib_centerDistance
        );


        // 计算标定参数
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        double rms = 0.0;
        std::vector<std::vector<cv::Point2f>> imageCorrdiantes;

        for(int i = 0; i < sortedCircleCenter.size(); i ++){
            std::vector<cv::Point2f> singleCorrdiantes;
            for (int j = 0; j < sortedCircleCenter[i].size(); j++){
                
                cv::Point2f p  = sortedCircleCenter[i][j].center;
                singleCorrdiantes.push_back(p);

            }
            imageCorrdiantes.push_back(singleCorrdiantes);
        }

        rms = cv::calibrateCamera(worldCorrdianates, imageCorrdiantes, cv::Size(images[0].cols, images[0].rows),
        cameraMatrix, distCoeffs, rvecs, tvecs);
        logInfo("Calibration finished. RMS = " + std::to_string(rms));


        // 计算重投影误差 评估质量

        shutdownLogger();


    }

}
