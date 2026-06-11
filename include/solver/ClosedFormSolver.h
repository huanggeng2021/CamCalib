#pragma once

#include "core/CalibrationData.h"
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace camcalib::solver {

template<typename PointT>
Circle fitCircleToEdges(const std::vector<PointT>& points){
    const int pointCount = static_cast<int>(points.size());
    std::vector<cv::Point2d> edgePoints;
    edgePoints.reserve(points.size());
    for(const PointT& point : points){
        edgePoints.emplace_back(static_cast<double>(point.x), static_cast<double>(point.y));
    }

    if(pointCount < 3){
        Circle circle;
        circle.edge_points = edgePoints;
        circle.center = cv::Point2d(0.0, 0.0);
        circle.radius = 0.0;
        return circle;
    }

    Eigen::MatrixXd a(pointCount, 3);
    Eigen::MatrixXd b(pointCount, 1);

    for(int pointIndex = 0; pointIndex < pointCount; ++pointIndex){
        const double x = static_cast<double>(points[pointIndex].x);
        const double y = static_cast<double>(points[pointIndex].y);
        a(pointIndex, 0) = x;
        a(pointIndex, 1) = y;
        a(pointIndex, 2) = 1.0;
        b(pointIndex, 0) = x * x + y * y;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::MatrixXd u = svd.matrixU();
    const Eigen::MatrixXd v = svd.matrixV();
    const Eigen::VectorXd singularValues = svd.singularValues();

    Eigen::MatrixXd sigmaPseudoInverse = Eigen::MatrixXd::Zero(v.cols(), u.rows());
    constexpr double tolerance = 1e-10;
    for(int valueIndex = 0; valueIndex < singularValues.size(); ++valueIndex){
        if(singularValues(valueIndex) > tolerance){
            sigmaPseudoInverse(valueIndex, valueIndex) = 1.0 / singularValues(valueIndex);
        }
    }

    const Eigen::VectorXd params = v * sigmaPseudoInverse * u.transpose() * b;
    const double d = params(0);
    const double e = params(1);
    const double f = params(2);

    Circle circle;
    circle.edge_points = edgePoints;
    circle.center = cv::Point2d(d / 2.0, e / 2.0);
    const double radiusSquared = circle.center.x * circle.center.x + circle.center.y * circle.center.y + f;
    circle.radius = radiusSquared > 0.0 ? std::sqrt(radiusSquared) : 0.0;
    return circle;
}

std::vector<std::vector<cv::Point3f>> generateWorldCoordinates(
    int imageCount,
    int calibRows,
    int calibCols,
    double centerDistance
);

}  // namespace camcalib::solver
