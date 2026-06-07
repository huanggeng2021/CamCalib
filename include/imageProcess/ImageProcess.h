#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include "imageProcess/CannyDetecter.h"
#include <Eigen/Dense>

namespace camcalib {    

class ImageProcess{

public:

    struct Circle{   // 标定板圆的信息

        std::vector<cv::Point2i> edge_points; // 圆边缘上的点
        cv::Point2d center; // 圆心坐标
        double radius; // 圆半径

        double homo_row;
        double homo_col;
    };

    // 加载图像入口
    std::vector<cv::Mat> loadImages();

    // 标定入口 
    void runCalibrate();
       

private:

    struct MarkerPair{
        int first = -1;
        int second = -1;
    };

    struct MarkerDistanceInfo{
        MarkerPair nearest;
        MarkerPair farthest;
    };

    // 加载灰度图像
    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths);

    //------------- 图像预处理模块------------
    std::vector<cv::Mat> detecTheCircleEdge(const std::vector<cv::Mat>& images);
    std::vector<std::vector<cv::Point>> filterContours(const std::vector<std::vector<cv::Point>>& contours);
    std::vector<std::vector<std::vector<cv::Point2i>>> getAllEdgePoints(const std::vector<cv::Mat>& images);
    // 图像预处理 检测圆的边缘
    std::vector<std::vector<std::vector<cv::Point>>> runDetectEdge(const std::vector<cv::Mat>& images);

    // 亚像素边缘提取部分
    std::vector<std::vector<std::vector<cv::Point>>> runDetectSubPixelEdge(const std::vector<cv::Mat>& images);





    
    // 圆心拟合
    Circle getCircleCenter(const std::vector<cv::Point2i>& points);

    // 对圆心进行排序
    std::vector<std::vector<Circle>> sortCircleCenter(std::vector<std::vector<Circle>>& disorderedCenter);
    static std::vector<Circle> getBigMarkers(const std::vector<Circle>& circles);
    static MarkerDistanceInfo findNearestAndFarthestMarkers(const std::vector<Circle>& bigMarkers);
    static int findRemainingMarkerIndex(const MarkerDistanceInfo& distanceInfo);
    static bool isMarkerPairParallel(const std::vector<Circle>& bigMarkers, const MarkerDistanceInfo& distanceInfo);
    static std::vector<Circle> orderBigMarkers(
        const std::vector<Circle>& bigMarkers,
        const MarkerDistanceInfo& distanceInfo,
        int p3Index
    );

    //计算变换矩阵
    static std::vector<Eigen::Matrix3d> findHomography(const std::vector<std::vector<Circle>>& sortedCircleCenter);
    // 变换到理想坐标系下

    static std::vector<std::vector<Circle>> homographyCircleCenter(
        const std::vector<Eigen::Matrix3d>& homo,
        const std::vector<std::vector<Circle>>& unSortCircleCenter
    );

    static std::vector<std::vector<cv::Point3f>> generateWorldCoordinates(
        const int imageNum,
        const int calibRows,
        const int calibCols,
        const double centerDist
    );




    // 辅助调试函数
    static cv::Point2d applyHomography(
        const Eigen::Matrix3d& homography,
        const cv::Point2d& point
    );

    static Eigen::Matrix3d cvMatToEigenMat3d(const cv::Mat& mat);

    static void printHomographyValidation(
        int imageIndex,
        const std::vector<cv::Point2d>& idealCenter,
        const std::vector<Circle>& sortedCircleCenter,
        const Eigen::Matrix3d& customH,
        const Eigen::Matrix3d& cvH
    );

    static void printCircleList(
        const std::string& title,
        const std::vector<Circle>& circles
    );

    void showEdgeAndCircleCenters(
        const cv::Mat& binary, 
        const std::vector<std::vector<cv::Point>>& contours, 
        const std::vector<Circle>& centerPoints, 
        const std::string& windowName);

    void showSortedCircleCenters(
        const cv::Mat& image, 
        const std::vector<Circle>& sortedCenterPoints, 
        const std::string& windowName);

    // 使用单映性矩阵变换原始图像
    void showWarpedImage(
        const cv::Mat& image, 
        const Eigen::Matrix3d& homography, 
        const std::string& windowName);

    void drawCenterPoints(cv::Mat& image, const std::vector<Circle>& centerPoints);




};

    

}
