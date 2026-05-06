// 实现Canny边缘检测算法的类，包含构造函数和detect函数，以及一些辅助函数
#include <cmath>
#include "imageProcess/CannyDetecter.h"

namespace camcalib {

CannyDetecter::CannyDetecter(int kernel_size, double sigma, double low_threshold, double high_threshold) {
    kernel_size_ = kernel_size;
    sigma_ = sigma;
    low_threshold_ = low_threshold;
    high_threshold_ = high_threshold;
}

cv::Mat CannyDetecter::getConvolutionKernel(cv::Size kernel_size, double sigma) {
    // Implementation for generating convolution kernel
    // 生成卷积核
    cv::Mat kernel(kernel_size, CV_64FC1);

    int width = kernel_size.width;
    int height = kernel_size.height;

    int centerW = (width - 1) / 2;
    int centerH = (height - 1) / 2;

    double x, y;
    double sum = 0;

    for (int i = 0; i < height; i++) {
        x = std::pow(i - centerH, 2);

        for (int j = 0; j < width; j++) {
            y = std::pow(j - centerW, 2);

            double temp = std::exp(-(x + y) / (2 * std::pow(sigma, 2)));
            kernel.at<double>(i, j) = temp;
            sum += temp;
        }
    }
    // 归一化
    if (sum != 0.0) {
        kernel /= sum;
    }

    return kernel;

}


// 辅助函数：应用高斯模糊
cv::Mat CannyDetecter::applyGaussianBlur(const cv::Mat& image, const cv::Mat& kernel) {
    // Implementation for applying Gaussian blur
    cv::Mat img;
    image.convertTo(img, CV_64FC1);

    cv::Mat blurred(img.rows, img.cols, CV_64FC1);

    int radius = kernel.rows / 2;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            double sum = 0.0;

            for(int m = -radius; m <= radius; m++){        
                for(int n = -radius; n <= radius; n++){
                    int x = std::min(std::max(j + n, 0), img.cols - 1);   // 边界处理，复制边界
                    int y = std::min(std::max(i + m, 0), img.rows - 1);

                    // 注意opencv中的行、列 与数学中的x、y坐标是相反的. .at<>中的第一个参数是行索引，第二个参数是列索引
                    sum += img.at<double>(y, x) * kernel.at<double>(m + radius, n + radius);
                }
            }

            blurred.at<double>(i, j) = sum;
        }
    }

    return blurred;

}


// 辅助函数：计算图像梯度和方向 image是高斯模糊后的图像
void CannyDetecter::computeGradient(const cv::Mat& image, cv::Mat& gradient_magnitude, cv::Mat& gradient_direction){
            
    // step1: 定义Sobel算子
    cv::Mat Gx = (cv::Mat_<double>(3, 3) << 
                    -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);

    cv::Mat Gy = (cv::Mat_<double>(3, 3) << 
                    -1, -2, -1,
                    0, 0, 0,
                    1, 2, 1);

    // step2: 使用Sobel算子计算图像的梯度，分别在x和y方向上
    cv::Mat grad_x = cv::Mat::zeros(image.size(), CV_64FC1);
    cv::Mat grad_y = cv::Mat::zeros(image.size(), CV_64FC1);
    int radius = 1; // Sobel算子大小为3x3，半径为1
    
    int rows = image.rows;
    int cols = image.cols;  

    for (int i = 1; i < rows - 1; i++)
    {
        for(int j = 1; j < cols - 1; j++)
        {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for(int m = -radius; m <= radius; m++)
            {
                for(int n = -radius; n <= radius; n++)
                {
                    int x = j + n;
                    int y = i + m;

                    sum_x += image.at<double>(y, x) * Gx.at<double>(m + radius, n + radius);
                    sum_y += image.at<double>(y, x) * Gy.at<double>(m + radius, n + radius);
                }
            }

            grad_x.at<double>(i, j) = sum_x;
            grad_y.at<double>(i, j) = sum_y;

        }
    }

    gradient_magnitude = cv::Mat::zeros(image.size(), CV_64FC1);
    gradient_direction = cv::Mat::zeros(image.size(), CV_64FC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double gx = grad_x.at<double>(i, j);
            double gy = grad_y.at<double>(i, j);

            gradient_magnitude.at<double>(i, j) = std::sqrt(gx * gx + gy * gy);
            gradient_direction.at<double>(i, j) = std::atan2(gy, gx);
        }
    }


}


 // 辅助函数：非极大值抑制
cv::Mat CannyDetecter::nonMaximumSuppression(const cv::Mat& gradient_magnitude, const cv::Mat& gradient_direction){
    // Implementation for non-maximum suppression
    cv::Mat non_max_suppressed = cv::Mat::zeros(gradient_magnitude.size(), CV_64FC1);

    int rows = gradient_magnitude.rows;
    int cols = gradient_magnitude.cols;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double angle = gradient_direction.at<double>(i, j) * 180.0 / CV_PI; // 将弧度转换为角度
            if (angle < 0) {   // atan2返回的角度范围是[-180, 180]，将其调整到[0, 180]范围内
                angle += 180; // 将角度调整到[0, 180]范围内
            }

            double mag = gradient_magnitude.at<double>(i, j);
            double mag1, mag2;

            // 根据梯度方向选择比较的像素
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {   // 水平情况 选择水平左右两侧的像素进行比较
                mag1 = gradient_magnitude.at<double>(i, j + 1); // 水平右侧
                mag2 = gradient_magnitude.at<double>(i, j - 1); // 水平左侧
            } else if (angle >= 22.5 && angle < 67.5) {                                // 斜向情况 选择右上和左下的像素进行比较
                mag1 = gradient_magnitude.at<double>(i - 1, j + 1); // 右上   
                mag2 = gradient_magnitude.at<double>(i + 1, j - 1); // 左下
            } else if (angle >= 67.5 && angle < 112.5) {
                mag1 = gradient_magnitude.at<double>(i - 1, j); // 垂直上方
                mag2 = gradient_magnitude.at<double>(i + 1, j); // 垂直下方
            } else { // angle >= 112.5 && angle < 157.5
                mag1 = gradient_magnitude.at<double>(i - 1, j - 1); // 左上
                mag2 = gradient_magnitude.at<double>(i + 1, j + 1); // 右下
            }

            // 如果当前像素的梯度幅值大于比较的两个像素，则保留，否则抑制
            if (mag >= mag1 && mag >= mag2) {
                non_max_suppressed.at<double>(i, j) = mag;
            } else {
                non_max_suppressed.at<double>(i, j) = 0.0;   
            }   
        }
    }

    return non_max_suppressed;
}


// 辅助函数：双阈值处理和边缘连接
cv::Mat CannyDetecter::thresholdAndLinkEdges(const cv::Mat& non_max_suppressed){
    // Implementation for thresholding and edge linking
    cv::Mat edges = cv::Mat::zeros(non_max_suppressed.size(), CV_8UC1);

    int rows = non_max_suppressed.rows;
    int cols = non_max_suppressed.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double mag = non_max_suppressed.at<double>(i, j);

            if (mag >= high_threshold_) {
                edges.at<uchar>(i, j) = 255; // 强边缘
            } else if (mag >= low_threshold_) {
                edges.at<uchar>(i, j) = 128; // 弱边缘
            } else {
                edges.at<uchar>(i, j) = 0;   // 非边缘
            }
        }
    }

    // 边缘连接：将弱边缘连接到强边缘
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (edges.at<uchar>(i, j) == 128) { // 如果是弱边缘
                // 检查8邻域是否有强边缘
                if (edges.at<uchar>(i - 1, j - 1) == 255 || edges.at<uchar>(i - 1, j) == 255 || edges.at<uchar>(i - 1, j + 1) == 255 ||
                    edges.at<uchar>(i, j - 1) == 255 || edges.at<uchar>(i, j + 1) == 255 ||
                    edges.at<uchar>(i + 1, j - 1) == 255 || edges.at<uchar>(i + 1, j) == 255 || edges.at<uchar>(i + 1, j + 1) == 255) {
                    edges.at<uchar>(i, j) = 255; // 将弱边缘连接为强边缘
                } else {
                    edges.at<uchar>(i, j) = 0;   // 否则抑制为非边缘
                }
            }
        }
    }

    return edges;

}


// 执行Canny边缘检测算法，返回边缘图像
cv::Mat CannyDetecter::detect(const cv::Mat& image){
    // step1: 生成高斯模糊核
    cv::Mat kernel = getConvolutionKernel(cv::Size(kernel_size_, kernel_size_), sigma_);

    // step2: 应用高斯模糊
    cv::Mat blurred = applyGaussianBlur(image, kernel);

    // step3: 计算图像梯度和方向
    cv::Mat gradient_magnitude, gradient_direction;
    computeGradient(blurred, gradient_magnitude, gradient_direction);

    // step4: 非极大值抑制
    cv::Mat non_max_suppressed = nonMaximumSuppression(gradient_magnitude, gradient_direction);

    // step5: 双阈值处理和边缘连接
    cv::Mat edges = thresholdAndLinkEdges(non_max_suppressed);

    return edges;
}

}