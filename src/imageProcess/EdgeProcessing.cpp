#include "imageProcess/EdgeProcessing.h"

#include <algorithm>
#include <complex>
#include <opencv2/imgproc.hpp>

namespace camcalib::image {
namespace {

cv::Mat buildBinaryImage(const cv::Mat& image){
    cv::Mat grayImage;
    if(image.channels() == 1){
        grayImage = image;
    }else{
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    return binaryImage;
}

PixelContours filterCircularContours(const PixelContours& contours){
    PixelContours pointCountFilteredContours;

    for(const PixelContour& contour : contours){
        if(contour.size() > 20){
            pointCountFilteredContours.push_back(contour);
        }
    }

    PixelContours circularContours;
    for(const PixelContour& contour : pointCountFilteredContours){
        if(contour.empty()){
            continue;
        }

        int minX = contour[0].x;
        int maxX = contour[0].x;
        int minY = contour[0].y;
        int maxY = contour[0].y;

        for(const cv::Point& point : contour){
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
        }

        const int width = maxX - minX + 1;
        const int height = maxY - minY + 1;
        const int longAxis = std::max(width, height);
        const int shortAxis = std::min(width, height);
        const double axisRatio = static_cast<double>(longAxis) / static_cast<double>(shortAxis);

        if(axisRatio <= 1.5){
            circularContours.push_back(contour);
        }
    }

    return circularContours;
}

SubPixelContoursByImage refineEdgesToSubPixel(
    const std::vector<cv::Mat>& images,
    const PixelContoursByImage& pixelEdges,
    cv::Size kernelSize
){
    SubPixelContoursByImage subPixelEdges;
    subPixelEdges.reserve(images.size());
    const int radius = kernelSize.width / 2;

    for(size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex){
        const cv::Mat& grayImage = images[imageIndex];
        SubPixelContours imageSubPixelEdges;
        imageSubPixelEdges.reserve(pixelEdges[imageIndex].size());

        for(size_t contourIndex = 0; contourIndex < pixelEdges[imageIndex].size(); ++contourIndex){
            SubPixelContour contourSubPixelEdges;
            contourSubPixelEdges.reserve(pixelEdges[imageIndex][contourIndex].size());

            for(const cv::Point& pixelPoint : pixelEdges[imageIndex][contourIndex]){
                std::complex<double> a11(0.0, 0.0);
                double a20 = 0.0;

                for(int dy = -radius; dy <= radius; ++dy){
                    for(int dx = -radius; dx <= radius; ++dx){
                        const double x = static_cast<double>(dx) / radius;
                        const double y = static_cast<double>(dy) / radius;
                        if(x * x + y * y > 1.0){
                            continue;
                        }

                        const double intensity = static_cast<double>(
                            grayImage.at<uchar>(pixelPoint.y + dy, pixelPoint.x + dx)
                        );
                        a11 += intensity * std::complex<double>(x, -y);
                        a20 += intensity * (2.0 * x * x + 2.0 * y * y - 1.0);
                    }
                }

                a11 *= 2.0 / CV_PI;
                a20 *= 3.0 / CV_PI;

                const double absA11 = std::abs(a11);
                if(absA11 < 1e-6){
                    continue;
                }

                const double phi = std::atan2(a11.imag(), a11.real());
                double offsetScale = -2.0 * a20 / (3.0 * absA11);
                if(std::abs(offsetScale) > 1.0){
                    continue;
                }

                contourSubPixelEdges.emplace_back(
                    pixelPoint.x + offsetScale * radius * std::cos(phi),
                    pixelPoint.y + offsetScale * radius * std::sin(phi)
                );
            }

            imageSubPixelEdges.push_back(contourSubPixelEdges);
        }

        subPixelEdges.push_back(imageSubPixelEdges);
    }

    return subPixelEdges;
}

}  // namespace

PixelContoursByImage detectEdges(const std::vector<cv::Mat>& images){
    PixelContoursByImage allImageContours;
    allImageContours.reserve(images.size());

    for(const cv::Mat& image : images){
        cv::Mat binaryImage = buildBinaryImage(image);
        PixelContours contours;
        cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        allImageContours.push_back(filterCircularContours(contours));
    }

    return allImageContours;
}

SubPixelContoursByImage detectSubPixelEdges(
    const std::vector<cv::Mat>& images,
    const PixelContoursByImage& pixelEdges
){
    return refineEdgesToSubPixel(images, pixelEdges, cv::Size(9, 9));
}

}  // namespace camcalib::image
