#pragma once

// 该模块作为手动实现Canny边缘检测算法的类，主要包含以下成员函数：
#include <opencv2/opencv.hpp>


namespace camcalib {

    class CannyDetecter {
    public:
        // 构造函数，接受高斯模糊核大小和低高阈值作为参数
        CannyDetecter(int kernel_size = 5, double sigma_ = 1.0,double low_threshold = 50, double high_threshold = 150);

        // 执行Canny边缘检测算法，返回边缘图像
        cv::Mat detect(const cv::Mat& image);

    private:
        int kernel_size_;       // 高斯模糊核大小
        double sigma_;        // 高斯模糊的标准差 
        double low_threshold_;  // 低阈值
        double high_threshold_; // 高阈值

        // 辅助函数：生成高斯模糊核
        cv::Mat getConvolutionKernel(cv::Size kernel_size, double sigma);

        // 辅助函数：应用高斯模糊
        cv::Mat applyGaussianBlur(const cv::Mat& image,const cv::Mat& kernel);

        // 辅助函数：计算图像梯度和方向
        void computeGradient(const cv::Mat& image, cv::Mat& gradient_magnitude, cv::Mat& gradient_direction);

        // 辅助函数：非极大值抑制
        cv::Mat nonMaximumSuppression(const cv::Mat& gradient_magnitude, const cv::Mat& gradient_direction);

        // 辅助函数：双阈值处理和边缘连接
        cv::Mat thresholdAndLinkEdges(const cv::Mat& non_max_suppressed);
    };
}


/*  改进计划  与 opencv 对比
1. thresholdAndLinkEdges() 改成 BFS / DFS 连接弱边缘
2. detect() 加输入检查：空图、通道数
3. applyGaussianBlur() 从 kernel.rows / kernel.cols 获取半径
4. nonMaximumSuppression() 给 mag1、mag2 初始化
5. 构造函数检查 kernel_size、sigma、阈值是否合法

*/