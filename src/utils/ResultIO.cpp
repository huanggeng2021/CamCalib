#include "utils/ResultIO.h"

#include <Eigen/Dense>

namespace camcalib::utils {

bool prepareDebugOutputDirectory(const std::filesystem::path& debugRoot){
    try{
        std::filesystem::create_directories(debugRoot);
    }catch(const std::filesystem::filesystem_error&){
        return false;
    }
    return true;
}

bool saveDebugImage(const std::filesystem::path& outputPath, const cv::Mat& image){
    try{
        std::filesystem::create_directories(outputPath.parent_path());
    }catch(const std::filesystem::filesystem_error&){
        return false;
    }

    return cv::imwrite(outputPath.string(), image);
}

void drawCenterPoints(cv::Mat& image, const CircleSet& centerPoints){
    for(const Circle& circle : centerPoints){
        if(circle.radius <= 0.0){
            continue;
        }

        const cv::Point center(
            static_cast<int>(std::round(circle.center.x)),
            static_cast<int>(std::round(circle.center.y))
        );
        cv::circle(image, center, 4, cv::Scalar(255, 0, 0), -1);
        cv::drawMarker(image, center, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 16, 2);
    }
}

cv::Mat renderSortedCircleCenters(const cv::Mat& image, const CircleSet& sortedCenterPoints){
    cv::Mat display;
    if(image.channels() == 1){
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    }else{
        display = image.clone();
    }

    for(size_t circleIndex = 0; circleIndex < sortedCenterPoints.size(); ++circleIndex){
        const Circle& circle = sortedCenterPoints[circleIndex];
        if(circle.radius <= 0.0){
            continue;
        }

        const cv::Point center(
            static_cast<int>(std::round(circle.center.x)),
            static_cast<int>(std::round(circle.center.y))
        );

        cv::circle(display, center, static_cast<int>(std::round(circle.radius)), cv::Scalar(0, 255, 255), 2);
        cv::circle(display, center, 4, cv::Scalar(0, 0, 255), -1);
        cv::drawMarker(display, center, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 16, 2);

        cv::Point textPosition(center.x + 8, center.y - 8);
        textPosition.x = std::max(0, std::min(textPosition.x, display.cols - 40));
        textPosition.y = std::max(20, std::min(textPosition.y, display.rows - 5));

        cv::putText(
            display,
            std::to_string(circleIndex),
            textPosition,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );
    }

    return display;
}

void showSortedCircleCenters(
    const cv::Mat& image,
    const CircleSet& sortedCenterPoints,
    const std::string& windowName
){
    cv::Mat display = renderSortedCircleCenters(image, sortedCenterPoints);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1200, 1000);
    cv::imshow(windowName, display);
    cv::waitKey(0);
}

void showWarpedImage(
    const cv::Mat& image,
    const Eigen::Matrix3d& homography,
    const std::string& windowName
){
    cv::Mat display;
    if(image.channels() == 1){
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    }else{
        display = image.clone();
    }

    cv::Mat homographyCv(3, 3, CV_64F);
    for(int row = 0; row < 3; ++row){
        for(int col = 0; col < 3; ++col){
            homographyCv.at<double>(row, col) = homography(row, col);
        }
    }

    const cv::Mat inverseHomography = homographyCv.inv();
    const std::vector<cv::Point2f> sourceCorners = {
        {0.0f, 0.0f},
        {static_cast<float>(display.cols - 1), 0.0f},
        {static_cast<float>(display.cols - 1), static_cast<float>(display.rows - 1)},
        {0.0f, static_cast<float>(display.rows - 1)}
    };

    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(sourceCorners, warpedCorners, inverseHomography);

    float minX = warpedCorners[0].x;
    float maxX = warpedCorners[0].x;
    float minY = warpedCorners[0].y;
    float maxY = warpedCorners[0].y;
    for(const cv::Point2f& point : warpedCorners){
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -minX;
    translation.at<double>(1, 2) = -minY;

    cv::Mat warped;
    cv::warpPerspective(
        display,
        warped,
        translation * inverseHomography,
        cv::Size(
            std::max(1, static_cast<int>(std::ceil(maxX - minX))),
            std::max(1, static_cast<int>(std::ceil(maxY - minY)))
        ),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1200, 1000);
    cv::imshow(windowName, warped);
    cv::waitKey(0);
}

}  // namespace camcalib::utils
