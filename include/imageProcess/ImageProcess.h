#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include "imageProcess/CannyDetecter.h"
#include <Eigen/Dense>

namespace camcalib {    

    struct Circle{

            std::vector<cv::Point2i> edge_points; // 圆边缘上的点
            cv::Point2d center; // 圆心坐标
            double radius; // 圆半径
    };

    // 加载灰度图像
    std::vector<cv::Mat> loadGrayImage(const std::vector<std::string>& image_paths);

    std::vector<cv::Mat> loadImages();


    // 图像预处理   应设计为类的 private
    // 获取所有图像的边缘
    std::vector<std::vector<Circle>> runDetectEdge(const std::vector<cv::Mat>& images);
    std::vector<cv::Mat> detecTheCircleEdge(const std::vector<cv::Mat>& images);
    std::vector<std::vector<std::vector<cv::Point2i>>> getAllEdgePoints(const std::vector<cv::Mat>& images);
    std::vector<std::vector<cv::Point2i>> getEdgePoints(const cv::Mat& edgeImage);




    // 拟合
    Circle getCircleCenter(const std::vector<cv::Point2i>& points);

}