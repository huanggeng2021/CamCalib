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
    };

    // 加载图像入口
    std::vector<cv::Mat> loadImages();

   
    void runCalibrate();
       

private:

    // 加载灰度图像
    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths);

    //------------- 图像预处理模块------------

    std::vector<cv::Mat> detecTheCircleEdge(const std::vector<cv::Mat>& images);
    std::vector<std::vector<cv::Point>> filterContours(const std::vector<std::vector<cv::Point>>& contours);
    std::vector<std::vector<std::vector<cv::Point2i>>> getAllEdgePoints(const std::vector<cv::Mat>& images);
    // 图像预处理 检测圆的边缘
    std::vector<std::vector<std::vector<cv::Point>>> runDetectEdge(const std::vector<cv::Mat>& images);

    void showEdgeAndCircleCenters(const cv::Mat& binary, const std::vector<std::vector<cv::Point>>& contours, const std::vector<Circle>& centerPoints, const std::string& windowName);
    void drawCenterPoints(cv::Mat& image, const std::vector<Circle>& centerPoints);

    // 拟合
    Circle getCircleCenter(const std::vector<cv::Point2i>& points);


};

    

}
