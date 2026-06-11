#include "imageProcess/ImageProcess.h"
#include "utils/Config.h"
#include <algorithm>
#include <complex>
#include <ctime>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>

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

    std::vector<double> ImageProcess::calculatePerImageReprojectionErrors(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs
    ){
        std::vector<double> perImageErrors;
        perImageErrors.reserve(imagePoints.size());

        for(size_t i = 0; i < imagePoints.size(); ++i){
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(
                objectPoints[i],
                rvecs[i],
                tvecs[i],
                cameraMatrix,
                distCoeffs,
                projectedPoints
            );

            double squaredErrorSum = 0.0;
            for(size_t j = 0; j < imagePoints[i].size(); ++j){
                const cv::Point2f diff = imagePoints[i][j] - projectedPoints[j];
                squaredErrorSum += static_cast<double>(diff.dot(diff));
            }

            const double rmsError = imagePoints[i].empty()
                ? 0.0
                : std::sqrt(squaredErrorSum / static_cast<double>(imagePoints[i].size()));

            perImageErrors.push_back(rmsError);
        }

        return perImageErrors;
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



    std::vector<std::vector<cv::Point>> ImageProcess::filterCircularContours(const std::vector<std::vector<cv::Point>>& contours){

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



    std::vector<std::vector<std::vector<cv::Point>>> ImageProcess::detectEdges(const std::vector<cv::Mat>& images){

        // 使用阈值分割的方式提取边缘点

        std::vector<std::vector<std::vector<cv::Point>>> allImageContours;
        allImageContours.reserve(images.size());

        for(size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex){

            cv::Mat binary = buildBinaryImage(images[imageIndex]);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            std::vector<std::vector<cv::Point>> filteredContours = filterCircularContours(contours);

            allImageContours.push_back(filteredContours);
        }
        

        return allImageContours;
    }


    std::vector<std::vector<std::vector<cv::Point2d>>> ImageProcess::refineEdgesToSubPixel(
        const std::vector<cv::Mat>& images, 
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        cv::Size kernelSize
    ){

        std::vector<std::vector<std::vector<cv::Point2d>>> subPixelEdges;
        subPixelEdges.reserve(images.size());
        const int radius = kernelSize.width / 2;

        for(size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex){

            const cv::Mat& grayImage = images[imageIndex];

            std::vector<std::vector<cv::Point2d>> imageSubPixelEdges;
            imageSubPixelEdges.reserve(pixelEdges[imageIndex].size());

            for(size_t contourIndex = 0; contourIndex < pixelEdges[imageIndex].size(); ++contourIndex){

                std::vector<cv::Point2d> contourSubPixelEdges;
                contourSubPixelEdges.reserve(pixelEdges[imageIndex][contourIndex].size());
                for(size_t edgeIndex = 0; edgeIndex < pixelEdges[imageIndex][contourIndex].size(); ++edgeIndex){

                    cv::Point pixelPoint = pixelEdges[imageIndex][contourIndex][edgeIndex];

                    std::complex<double> A11(0.0, 0.0);   // complex  表示复数形式
                    double A20 = 0.0;

                    for(int dy = -radius; dy <= radius; ++dy){

                        for(int dx = -radius; dx <= radius; ++dx){

                            double x = static_cast<double>(dx) / radius;
                            double y = static_cast<double>(dy) / radius;

                            if (x * x +  y * y > 1.0)
                            {
                                continue;
                            }

                            double intensity = static_cast<double>(grayImage.at<uchar>(pixelPoint.y + dy, pixelPoint.x + dx));

                            A11 += intensity * std::complex<double>(x, -y);
                            A20 += intensity * (2.0 * x * x + 2.0 * y * y - 1.0);

                            
                        }

                    }

                    A11 *= 2.0 / CV_PI;
                    A20 *= 3.0 / CV_PI;

                    double absA11 = std::abs(A11);

                    if(absA11 < 1e-6)
                        continue;

                    double phi = std::atan2(A11.imag(), A11.real());

                    double l = 2.0 * A20 / (3.0 * absA11);
                    l = -l;

                    if(std::abs(l) > 1.0)
                        continue;

                    double xs = pixelPoint.x + l * radius * std::cos(phi);
                    double ys = pixelPoint.y + l * radius * std::sin(phi);

                    contourSubPixelEdges.emplace_back(xs, ys);
                    

                }
                imageSubPixelEdges.push_back(contourSubPixelEdges);

                
            }
            subPixelEdges.push_back(imageSubPixelEdges);
        }

        return subPixelEdges;


    }

    std::vector<std::vector<std::vector<cv::Point2d>>> ImageProcess::detectSubPixelEdges(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges
    ){

        return refineEdgesToSubPixel(images, pixelEdges, cv::Size(9, 9));
    }



    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::sortMarkerCenters(std::vector<std::vector<Circle>>& unsortedCenters){

        std::vector<std::vector<ImageProcess::Circle>> sortedAllImages;
        const int imageCount = static_cast<int>(unsortedCenters.size());

        for(int imageIndex = 0; imageIndex < imageCount; ++imageIndex){

            std::vector<ImageProcess::Circle> bigMarkers = getBigMarkers(unsortedCenters[imageIndex]);

            if(bigMarkers.size() < 5){
                logError("Image " + std::to_string(imageIndex) + " does not have enough marker circles.");
                sortedAllImages.push_back({});
                continue;
            }

            MarkerDistanceInfo distanceInfo = findNearestAndFarthestMarkers(bigMarkers);
            int p3Index = findRemainingMarkerIndex(distanceInfo);

            if(p3Index == -1){
                logError("Image " + std::to_string(imageIndex) + " failed to find p3 marker.");
                sortedAllImages.push_back({});
                continue;
            }

            if(!isMarkerPairParallel(bigMarkers, distanceInfo)){
                logError("Image " + std::to_string(imageIndex) + " marker ordering failed because marker pairs are not parallel.");
                sortedAllImages.push_back({});
                continue;
            }

            std::vector<ImageProcess::Circle> orderedMarkers = orderBigMarkers(bigMarkers, distanceInfo, p3Index);
            sortedAllImages.push_back(orderedMarkers);

        }


        return sortedAllImages;

    }


    std::vector<Eigen::Matrix3d> ImageProcess::findHomography(const std::vector<std::vector<ImageProcess::Circle>>& sortedCircleCenter){

        const int imageCount = static_cast<int>(sortedCircleCenter.size());

        std::vector<Eigen::Matrix3d> homographyVec;
        homographyVec.reserve(imageCount);

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

       

        for(int imageIndex = 0; imageIndex < imageCount; ++imageIndex){

            Eigen::MatrixXd A(2 * 5, 9);
            A.setZero();
            std::vector<cv::Point2d> imageCenters;
            imageCenters.reserve(5);

            for (int iCircle = 0; iCircle < 5; iCircle++){

                double X = idealCenter[iCircle].x;
                double Y = idealCenter[iCircle].y;
                
                double u = sortedCircleCenter[imageIndex][iCircle].center.x;
                double v = sortedCircleCenter[imageIndex][iCircle].center.y;
                imageCenters.push_back(sortedCircleCenter[imageIndex][iCircle].center);

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

            homographyVec.push_back(H);


        }

        return homographyVec;

    }


    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::sortBoardCirclesByHomography(
        const std::vector<Eigen::Matrix3d>& homographies,
        const std::vector<std::vector<ImageProcess::Circle>>& unsortedCircleCenters
    ){

        const int imageCount = static_cast<int>(unsortedCircleCenters.size());

        std::vector<std::vector<ImageProcess::Circle>> sortedCircleCenters;
        sortedCircleCenters.reserve(imageCount);

    
        if(unsortedCircleCenters.size() != homographies.size()){
            return {};
        }

        for(int imageIndex = 0; imageIndex < imageCount; ++imageIndex){

            Eigen::MatrixXd inverseHomography = homographies[imageIndex].inverse();
            std::vector<Circle> homographySortedCircles;

            for(size_t circleIndex = 0; circleIndex < unsortedCircleCenters[imageIndex].size(); ++circleIndex){

                cv::Point2d imagePoint = unsortedCircleCenters[imageIndex][circleIndex].center;
                Circle circle = unsortedCircleCenters[imageIndex][circleIndex];

                Eigen::Vector3d p_img;
                p_img << imagePoint.x, imagePoint.y, 1.0;

                Eigen::Vector3d p_ideal_h = inverseHomography * p_img;

                if (std::abs(p_ideal_h(2)) < 1e-12)
                {
                    throw std::runtime_error("Homogeneous coordinate is too close to zero.");
                }

                double X = p_ideal_h(0) / p_ideal_h(2);
                double Y = p_ideal_h(1) / p_ideal_h(2);

                // OpenCV 点坐标使用 (x, y) = (col, row)，排序时要按 row 再按 col。
                circle.homo_col = X;
                circle.homo_row = Y;
                
                homographySortedCircles.push_back(circle);
            }

            // 排序
            // 先按行 homo_row 从小到大，再按列 homo_col 从小到大
            double dCenterDistance = 15.0;
            double rowTolerance = dCenterDistance * 0.5;

            std::sort(homographySortedCircles.begin(), homographySortedCircles.end(),
                [rowTolerance](const Circle& a, const Circle& b)
                {
                    if (std::abs(a.homo_row - b.homo_row) > rowTolerance)
                    {
                        return a.homo_row < b.homo_row;
                    }

                    return a.homo_col < b.homo_col;
                });

            sortedCircleCenters.push_back(homographySortedCircles);
        }

        return sortedCircleCenters;

    }

    
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

    template<typename pointT>
    cv::Mat ImageProcess::renderEdgeAndCircleCenters(
        const cv::Mat& image,
        const std::vector<std::vector<pointT>>& edges,
        const std::vector<Circle>& centerPoints
    ){

        cv::Mat display;
        if(image.channels() == 1){
            cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
        }else{
            display = image.clone();
        }

        cv::Mat overlay = display.clone();
        constexpr bool isSubPixel = std::is_floating_point_v<typename pointT::value_type>;
        constexpr int pointShift = isSubPixel ? 8 : 0;
        constexpr int pointScale = 1 << pointShift;

        for(size_t j = 0; j < edges.size(); ++j){
            if(edges[j].empty()){
                continue;
            }

            cv::Point2d meanPoint(0.0, 0.0);
            std::vector<cv::Point> polyline;
            polyline.reserve(edges[j].size());

            for(const pointT& point : edges[j]){
                meanPoint += cv::Point2d(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y)
                );
                polyline.emplace_back(
                    static_cast<int>(std::round(static_cast<double>(point.x) * pointScale)),
                    static_cast<int>(std::round(static_cast<double>(point.y) * pointScale))
                );
            }

            for(size_t k = 1; k < polyline.size(); ++k){
                cv::line(
                    overlay,
                    polyline[k - 1],
                    polyline[k],
                    cv::Scalar(0, 0, 255),
                    1,
                    cv::LINE_AA,
                    pointShift
                );
            }

            if(polyline.size() > 2){
                cv::line(
                    overlay,
                    polyline.back(),
                    polyline.front(),
                    cv::Scalar(0, 0, 255),
                    1,
                    cv::LINE_AA,
                    pointShift
                );
            }

            const size_t markerStep = std::max<size_t>(1, edges[j].size() / 24);
            for(size_t k = 0; k < edges[j].size(); k += markerStep){
                const pointT& point = edges[j][k];
                cv::circle(
                    overlay,
                    cv::Point(
                        static_cast<int>(std::round(static_cast<double>(point.x))),
                        static_cast<int>(std::round(static_cast<double>(point.y)))
                    ),
                    1,
                    cv::Scalar(80, 220, 255),
                    -1,
                    cv::LINE_AA
                );
            }

            meanPoint *= (1.0 / static_cast<double>(edges[j].size()));
            cv::putText(
                overlay,
                std::to_string(j),
                cv::Point(
                    static_cast<int>(std::round(meanPoint.x)),
                    std::max(15, static_cast<int>(std::round(meanPoint.y)) - 5)
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 255, 0),
                1,
                cv::LINE_AA
            );
        }

        cv::addWeighted(overlay, 0.75, display, 0.25, 0.0, display);
        drawCenterPoints(display, centerPoints);
        return display;
    }

    template cv::Mat ImageProcess::renderEdgeAndCircleCenters<cv::Point>(
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point>>& edges,
        const std::vector<Circle>& centerPoints
    );

    template cv::Mat ImageProcess::renderEdgeAndCircleCenters<cv::Point2d>(
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point2d>>& edges,
        const std::vector<Circle>& centerPoints
    );

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

    template<typename pointT>
    bool ImageProcess::saveEdgesToText(
        const std::filesystem::path& outputPath,
        const std::vector<std::vector<pointT>>& edges
    ){

        try{
            std::filesystem::create_directories(outputPath.parent_path());
        }catch(const std::filesystem::filesystem_error& e){
            logError("Failed to create debug directory: " + std::string(e.what()));
            return false;
        }

        std::ofstream outFile(outputPath);
        if(!outFile.is_open()){
            logError("Failed to open edge output file: " + outputPath.string());
            return false;
        }

        outFile << std::fixed << std::setprecision(6);
        outFile << "# contour_index point_index x y\n";

        for(size_t contourIndex = 0; contourIndex < edges.size(); ++contourIndex){
            const std::vector<pointT>& contour = edges[contourIndex];
            for(size_t pointIndex = 0; pointIndex < contour.size(); ++pointIndex){
                outFile
                    << contourIndex << ' '
                    << pointIndex << ' '
                    << static_cast<double>(contour[pointIndex].x) << ' '
                    << static_cast<double>(contour[pointIndex].y) << '\n';
            }
        }

        return true;
    }

    template bool ImageProcess::saveEdgesToText<cv::Point>(
        const std::filesystem::path& outputPath,
        const std::vector<std::vector<cv::Point>>& edges
    );

    template bool ImageProcess::saveEdgesToText<cv::Point2d>(
        const std::filesystem::path& outputPath,
        const std::vector<std::vector<cv::Point2d>>& edges
    );


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

    template<typename pointT>
    ImageProcess::Circle ImageProcess::fitCircleToEdges(const std::vector<pointT>& points){

        const int N = static_cast<int>(points.size());
        std::vector<cv::Point2d> edgePoints;
        edgePoints.reserve(points.size());
        for(const pointT& point : points){
            edgePoints.emplace_back(static_cast<double>(point.x), static_cast<double>(point.y));
        }

        if(N < 3){
            Circle c;
            c.edge_points = edgePoints;
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
        c.edge_points = edgePoints;
        c.center = cv::Point2d(D / 2.0, E / 2.0);   // 圆心 xy  与cv::point2d中的 xy并不一致
        double radius_sq = c.center.x * c.center.x + c.center.y * c.center.y + F;
        c.radius = radius_sq > 0.0 ? std::sqrt(radius_sq) : 0.0;

        return c;

    }

    template ImageProcess::Circle ImageProcess::fitCircleToEdges<cv::Point>(
        const std::vector<cv::Point>& points
    );

    template ImageProcess::Circle ImageProcess::fitCircleToEdges<cv::Point2d>(
        const std::vector<cv::Point2d>& points
    );

    bool ImageProcess::prepareDebugOutputDirectory(const std::filesystem::path& debugRoot){
        try{
            std::filesystem::create_directories(debugRoot);
        }catch(const std::filesystem::filesystem_error& e){
            logError("Failed to create debug output directory: " + std::string(e.what()));
            return false;
        }

        logInfo("Debug images will be saved to: " + debugRoot.string());
        return true;
    }

    void ImageProcess::detectEdgesStage(
        const std::vector<cv::Mat>& images,
        std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        std::vector<std::vector<std::vector<cv::Point2d>>>& subPixelEdges
    ){
        pixelEdges = detectEdges(images);
        logInfo("Edge detection completed for " + std::to_string(pixelEdges.size()) + " images.");

        subPixelEdges = detectSubPixelEdges(images, pixelEdges);
        logInfo("Subpixel edge refinement completed for " + std::to_string(subPixelEdges.size()) + " images.");
    }

    void ImageProcess::saveDetectedEdgeArtifacts(
        const std::filesystem::path& imageDebugDir,
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point>>& pixelEdges,
        const std::vector<std::vector<cv::Point2d>>& subPixelEdges,
        const std::vector<Circle>& fittedCircles
    ){
        saveEdgesToText(imageDebugDir / "00_pixel_edges.txt", pixelEdges);
        saveEdgesToText(imageDebugDir / "01_subpixel_edges.txt", subPixelEdges);
        saveDebugImage(
            imageDebugDir / "02_detected_edges.png",
            renderEdgeAndCircleCenters(image, subPixelEdges, fittedCircles)
        );
        saveDebugImage(
            imageDebugDir / "03_fitted_centers.png",
            renderSortedCircleCenters(image, fittedCircles)
        );
    }

    std::vector<std::vector<ImageProcess::Circle>> ImageProcess::fitCircleCentersStage(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        const std::vector<std::vector<std::vector<cv::Point2d>>>& subPixelEdges,
        bool shouldSaveDebugImages,
        const std::filesystem::path& debugRoot
    ){
        std::vector<std::vector<Circle>> allFittedCircles;
        allFittedCircles.reserve(subPixelEdges.size());

        for(size_t imageIndex = 0; imageIndex < subPixelEdges.size(); ++imageIndex){
            std::vector<Circle> fittedCircles;
            fittedCircles.reserve(subPixelEdges[imageIndex].size());

            for(const std::vector<cv::Point2d>& contour : subPixelEdges[imageIndex]){
                fittedCircles.push_back(fitCircleToEdges<cv::Point2d>(contour));
            }

            allFittedCircles.push_back(fittedCircles);
            logInfo(
                "Image " + std::to_string(imageIndex) +
                ": contours=" + std::to_string(pixelEdges[imageIndex].size()) +
                ", fitted_centers=" + std::to_string(fittedCircles.size())
            );

            if(shouldSaveDebugImages){
                std::ostringstream imageFolderName;
                imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
                const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();
                saveDetectedEdgeArtifacts(
                    imageDebugDir,
                    images[imageIndex],
                    pixelEdges[imageIndex],
                    subPixelEdges[imageIndex],
                    fittedCircles
                );
                logInfo("Saved edge and fitting debug outputs for image " + std::to_string(imageIndex) + ".");
            }
        }

        return allFittedCircles;
    }

    void ImageProcess::saveSortedMarkerArtifacts(
        const std::filesystem::path& debugRoot,
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedMarkerCenters
    ){
        for(size_t imageIndex = 0; imageIndex < sortedMarkerCenters.size() && imageIndex < images.size(); ++imageIndex){
            if(sortedMarkerCenters[imageIndex].empty()){
                logError("Image " + std::to_string(imageIndex) + " has no sorted marker output.");
                continue;
            }

            std::ostringstream imageFolderName;
            imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
            const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

            saveDebugImage(
                imageDebugDir / "04_sorted_markers.png",
                renderSortedCircleCenters(images[imageIndex], sortedMarkerCenters[imageIndex])
            );
            logInfo("Saved sorted marker debug image for image " + std::to_string(imageIndex) + ".");
        }
    }

    void ImageProcess::saveSortedBoardArtifacts(
        const std::filesystem::path& debugRoot,
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    ){
        for(size_t imageIndex = 0; imageIndex < sortedBoardCircles.size() && imageIndex < images.size(); ++imageIndex){
            if(sortedBoardCircles[imageIndex].empty()){
                logError("Image " + std::to_string(imageIndex) + " has no final sorted board output.");
                continue;
            }

            std::ostringstream imageFolderName;
            imageFolderName << "image_" << std::setw(3) << std::setfill('0') << imageIndex;
            const std::filesystem::path imageDebugDir = debugRoot / imageFolderName.str();

            saveDebugImage(
                imageDebugDir / "05_sorted_board.png",
                renderSortedCircleCenters(images[imageIndex], sortedBoardCircles[imageIndex])
            );
            logInfo("Saved final sorted board debug image for image " + std::to_string(imageIndex) + ".");
        }
    }

    void ImageProcess::showDebugWindows(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedMarkerCenters,
        const std::vector<Eigen::Matrix3d>& homographies,
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    ){
        for(size_t imageIndex = 0; imageIndex < sortedMarkerCenters.size() && imageIndex < images.size(); ++imageIndex){
            if(sortedMarkerCenters[imageIndex].empty()){
                continue;
            }

            showSortedCircleCenters(
                images[imageIndex],
                sortedMarkerCenters[imageIndex],
                "Sorted Circle Centers " + std::to_string(imageIndex)
            );
        }

        for(size_t imageIndex = 0; imageIndex < sortedBoardCircles.size() &&
                                   imageIndex < images.size() &&
                                   imageIndex < homographies.size(); ++imageIndex){
            if(sortedBoardCircles[imageIndex].empty()){
                continue;
            }

            showWarpedImage(
                images[imageIndex],
                homographies[imageIndex],
                "Warped Image " + std::to_string(imageIndex)
            );

            showSortedCircleCenters(
                images[imageIndex],
                sortedBoardCircles[imageIndex],
                "Sorted Board Circles " + std::to_string(imageIndex)
            );
        }
    }

    std::vector<std::vector<cv::Point2f>> ImageProcess::collectImagePoints(
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    ){
        std::vector<std::vector<cv::Point2f>> imagePoints;
        imagePoints.reserve(sortedBoardCircles.size());

        for(const std::vector<Circle>& imageCircles : sortedBoardCircles){
            std::vector<cv::Point2f> imageCircleCenters;
            imageCircleCenters.reserve(imageCircles.size());
            for(const Circle& circle : imageCircles){
                imageCircleCenters.emplace_back(circle.center);
            }
            imagePoints.push_back(imageCircleCenters);
        }

        return imagePoints;
    }


    void ImageProcess::runCalibrate(){

        const std::string configPath = "config/calib_config.yaml";

        CaliConfig config;
        if(!ConfigReader::readConfig(configPath, config)){
            std::cerr << "Failed to read config file: " << configPath << std::endl;
            return;
        }

        initializeLogger(config);
        logInfo("Calibration started.");
        logInfo("Current working directory: " + std::filesystem::current_path().string());
        logInfo("Loaded config file: " + configPath);

        const bool shouldSaveDebugImages = config.debug_save_images;
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

        const std::filesystem::path debugRoot = config.debug_output_dir;
        if(shouldSaveDebugImages && !prepareDebugOutputDirectory(debugRoot)){
            shutdownLogger();
            return;
        }

        std::vector<std::vector<std::vector<cv::Point>>> pixelEdges;
        std::vector<std::vector<std::vector<cv::Point2d>>> subPixelEdges;
        detectEdgesStage(images, pixelEdges, subPixelEdges);

        std::vector<std::vector<Circle>> fittedCircleCenters = fitCircleCentersStage(
            images,
            pixelEdges,
            subPixelEdges,
            shouldSaveDebugImages,
            debugRoot
        );

        std::vector<std::vector<Circle>> sortedMarkerCenters = sortMarkerCenters(fittedCircleCenters);
        logInfo("Marker sorting completed.");

        if(shouldSaveDebugImages){
            saveSortedMarkerArtifacts(debugRoot, images, sortedMarkerCenters);
        }

        std::vector<Eigen::Matrix3d> homographies = findHomography(sortedMarkerCenters);
        logInfo("Homography computation completed.");

        std::vector<std::vector<ImageProcess::Circle>> sortedBoardCircles =
            sortBoardCirclesByHomography(homographies, fittedCircleCenters);
        logInfo("Full board sorting completed.");

        if(shouldSaveDebugImages){
            saveSortedBoardArtifacts(debugRoot, images, sortedBoardCircles);
        }

        if(shouldShowDebugWindows){
            showDebugWindows(images, sortedMarkerCenters, homographies, sortedBoardCircles);
        }

        std::vector<std::vector<cv::Point3f>> worldCoordinates = generateWorldCoordinates(
            static_cast<int>(sortedBoardCircles.size()),
            config.calib_rows,
            config.calib_cols,
            config.calib_centerDistance
        );

        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        std::vector<cv::Mat> rotationVectors;
        std::vector<cv::Mat> translationVectors;
        const std::vector<std::vector<cv::Point2f>> imagePoints = collectImagePoints(sortedBoardCircles);

        const double rms = cv::calibrateCamera(
            worldCoordinates,
            imagePoints,
            cv::Size(images[0].cols, images[0].rows),
            cameraMatrix,
            distCoeffs,
            rotationVectors,
            translationVectors
        );
        logInfo("Calibration finished. RMS = " + std::to_string(rms));

        const std::vector<double> perImageErrors = calculatePerImageReprojectionErrors(
            worldCoordinates,
            imagePoints,
            rotationVectors,
            translationVectors,
            cameraMatrix,
            distCoeffs
        );

        for(size_t imageIndex = 0; imageIndex < perImageErrors.size(); ++imageIndex){
            std::ostringstream message;
            message << "Image " << imageIndex << " reprojection error = "
                    << std::fixed << std::setprecision(6) << perImageErrors[imageIndex]
                    << " pixels";
            logInfo(message.str());
            std::cout << message.str() << std::endl;
        }

        shutdownLogger();
    }

}
