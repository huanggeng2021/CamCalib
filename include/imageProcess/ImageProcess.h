#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include "utils/Config.h"
#include <Eigen/Dense>

namespace camcalib {    

class ImageProcess{

public:

    struct Circle{   // 标定板圆的信息

        std::vector<cv::Point2d> edge_points; // 圆边缘上的点
        cv::Point2d center; // 圆心坐标
        double radius; // 圆半径

        double homo_row;
        double homo_col;
    };

    // 加载图像入口
    std::vector<cv::Mat> loadImages(const CaliConfig& config);

    // 标定入口 
    void runCalibrate();
       

private:
    bool log_enabled_ = false;
    std::ofstream log_stream_;

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
    std::vector<std::vector<cv::Point>> filterCircularContours(const std::vector<std::vector<cv::Point>>& contours);
    std::vector<std::vector<std::vector<cv::Point>>> detectEdges(const std::vector<cv::Mat>& images);

    // 亚像素边缘提取部分
    std::vector<std::vector<std::vector<cv::Point2d>>> detectSubPixelEdges(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges
    );
    
    // 计算亚像素坐标偏移
    static std::vector<std::vector<std::vector<cv::Point2d>>> refineEdgesToSubPixel(
        const std::vector<cv::Mat>& images, 
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        cv::Size kernelSize
    );
    
    // 圆心拟合
    template<typename pointT>
    Circle fitCircleToEdges(const std::vector<pointT>& points);

    // 对圆心进行排序
    std::vector<std::vector<Circle>> sortMarkerCenters(std::vector<std::vector<Circle>>& unsortedCenters);
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

    static std::vector<std::vector<Circle>> sortBoardCirclesByHomography(
        const std::vector<Eigen::Matrix3d>& homographies,
        const std::vector<std::vector<Circle>>& unsortedCircleCenters
    );

    static std::vector<std::vector<cv::Point3f>> generateWorldCoordinates(
        const int imageNum,
        const int calibRows,
        const int calibCols,
        const double centerDist
    );

    static std::vector<double> calculatePerImageReprojectionErrors(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs
    );


    // 辅助调试函数
    static cv::Mat buildBinaryImage(const cv::Mat& image);

    template<typename pointT>
    static cv::Mat renderEdgeAndCircleCenters(
        const cv::Mat& image,
        const std::vector<std::vector<pointT>>& edges,
        const std::vector<Circle>& centerPoints
    );

    static cv::Mat renderSortedCircleCenters(
        const cv::Mat& image,
        const std::vector<Circle>& sortedCenterPoints
    );

    bool saveDebugImage(
        const std::filesystem::path& outputPath,
        const cv::Mat& image
    );

    template<typename pointT>
    bool saveEdgesToText(
        const std::filesystem::path& outputPath,
        const std::vector<std::vector<pointT>>& edges
    );

    void initializeLogger(const CaliConfig& config);
    void shutdownLogger();
    void logMessage(const std::string& level, const std::string& message);
    void logInfo(const std::string& message);
    void logError(const std::string& message);

    void showSortedCircleCenters(
        const cv::Mat& image, 
        const std::vector<Circle>& sortedCenterPoints, 
        const std::string& windowName);

    // 使用单映性矩阵变换原始图像
    void showWarpedImage(
        const cv::Mat& image, 
        const Eigen::Matrix3d& homography, 
        const std::string& windowName);

    static void drawCenterPoints(cv::Mat& image, const std::vector<Circle>& centerPoints);

    bool prepareDebugOutputDirectory(const std::filesystem::path& debugRoot);

    void detectEdgesStage(
        const std::vector<cv::Mat>& images,
        std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        std::vector<std::vector<std::vector<cv::Point2d>>>& subPixelEdges
    );

    std::vector<std::vector<Circle>> fitCircleCentersStage(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<std::vector<cv::Point>>>& pixelEdges,
        const std::vector<std::vector<std::vector<cv::Point2d>>>& subPixelEdges,
        bool shouldSaveDebugImages,
        const std::filesystem::path& debugRoot
    );

    void saveDetectedEdgeArtifacts(
        const std::filesystem::path& imageDebugDir,
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point>>& pixelEdges,
        const std::vector<std::vector<cv::Point2d>>& subPixelEdges,
        const std::vector<Circle>& fittedCircles
    );

    void saveSortedMarkerArtifacts(
        const std::filesystem::path& debugRoot,
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedMarkerCenters
    );

    void saveSortedBoardArtifacts(
        const std::filesystem::path& debugRoot,
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    );

    void showDebugWindows(
        const std::vector<cv::Mat>& images,
        const std::vector<std::vector<Circle>>& sortedMarkerCenters,
        const std::vector<Eigen::Matrix3d>& homographies,
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    );

    static std::vector<std::vector<cv::Point2f>> collectImagePoints(
        const std::vector<std::vector<Circle>>& sortedBoardCircles
    );




};

    

}
