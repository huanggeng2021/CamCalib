#include <filesystem>
#include <iostream>
#include <vector>
#include "utils/Config.h"
#include "imageProcess/ImageLoader.h"
#include "imageProcess/CannyDetecter.h"

int main() {
   
    std::vector<cv::Mat> grayImage = camcalib::loadImages();

    camcalib::CannyDetecter canny(5, 1.0, 50, 100);

    cv::Mat edges = canny.detect(grayImage[0]);
 
  
    // 创建窗口并设置窗口大小
    cv::namedWindow("Canny Edges", cv::WINDOW_NORMAL);
    cv::resizeWindow("Canny Edges", std::min(800, grayImage[0].cols * 2), std::min(600, grayImage[0].rows * 2));
    cv::imshow("Canny Edges", edges);
    
    std::cout << "Displaying edges image. Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}