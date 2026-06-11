#include "solver/ClosedFormSolver.h"

namespace camcalib::solver {

std::vector<std::vector<cv::Point3f>> generateWorldCoordinates(
    int imageCount,
    int calibRows,
    int calibCols,
    double centerDistance
){
    std::vector<std::vector<cv::Point3f>> worldPointsByImage;
    worldPointsByImage.reserve(imageCount);

    std::vector<cv::Point3f> worldPoints;
    worldPoints.reserve(calibRows * calibCols);

    const double centerOffsetX = (calibCols / 2) * centerDistance;
    const double centerOffsetY = (calibRows / 2) * centerDistance;

    for(int rowIndex = 0; rowIndex < calibRows; ++rowIndex){
        for(int colIndex = 0; colIndex < calibCols; ++colIndex){
            worldPoints.emplace_back(
                colIndex * centerDistance - centerOffsetX,
                rowIndex * centerDistance - centerOffsetY,
                0.0
            );
        }
    }

    worldPointsByImage.resize(imageCount, worldPoints);
    return worldPointsByImage;
}

}  // namespace camcalib::solver
