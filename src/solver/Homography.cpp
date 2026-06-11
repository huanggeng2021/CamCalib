#include "solver/Homography.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

namespace camcalib::solver {
namespace {

struct MarkerPair {
    int first = -1;
    int second = -1;
};

struct MarkerDistanceInfo {
    MarkerPair nearest;
    MarkerPair farthest;
};

std::vector<Circle> getBigMarkers(const CircleSet& circles){
    std::vector<Circle> sortedCircles = circles;
    std::sort(
        sortedCircles.begin(),
        sortedCircles.end(),
        [](const Circle& left, const Circle& right){ return left.radius > right.radius; }
    );

    const size_t markerCount = std::min<size_t>(5, sortedCircles.size());
    return std::vector<Circle>(sortedCircles.begin(), sortedCircles.begin() + markerCount);
}

MarkerDistanceInfo findNearestAndFarthestMarkers(const CircleSet& bigMarkers){
    MarkerDistanceInfo distanceInfo;
    double minDistance = std::numeric_limits<double>::max();
    double maxDistance = std::numeric_limits<double>::lowest();

    for(int firstIndex = 0; firstIndex < 5; ++firstIndex){
        for(int secondIndex = firstIndex + 1; secondIndex < 5; ++secondIndex){
            const double dx = bigMarkers[firstIndex].center.x - bigMarkers[secondIndex].center.x;
            const double dy = bigMarkers[firstIndex].center.y - bigMarkers[secondIndex].center.y;
            const double distance = std::sqrt(dx * dx + dy * dy);

            if(distance < minDistance){
                minDistance = distance;
                distanceInfo.nearest = {firstIndex, secondIndex};
            }

            if(distance > maxDistance){
                maxDistance = distance;
                distanceInfo.farthest = {firstIndex, secondIndex};
            }
        }
    }

    return distanceInfo;
}

int findRemainingMarkerIndex(const MarkerDistanceInfo& distanceInfo){
    for(int markerIndex = 0; markerIndex < 5; ++markerIndex){
        if(markerIndex != distanceInfo.nearest.first &&
           markerIndex != distanceInfo.nearest.second &&
           markerIndex != distanceInfo.farthest.first &&
           markerIndex != distanceInfo.farthest.second){
            return markerIndex;
        }
    }
    return -1;
}

bool isMarkerPairParallel(const CircleSet& bigMarkers, const MarkerDistanceInfo& distanceInfo){
    const cv::Point2d nearP1 = bigMarkers[distanceInfo.nearest.first].center;
    const cv::Point2d nearP2 = bigMarkers[distanceInfo.nearest.second].center;
    const cv::Point2d farP1 = bigMarkers[distanceInfo.farthest.first].center;
    const cv::Point2d farP2 = bigMarkers[distanceInfo.farthest.second].center;

    const cv::Point2d v1 = nearP2 - nearP1;
    const cv::Point2d v2 = farP2 - farP1;
    const Eigen::Vector2d vector1(v1.x, v1.y);
    const Eigen::Vector2d vector2(v2.x, v2.y);

    const double dot = vector1.dot(vector2);
    const double cross = vector1.x() * vector2.y() - vector1.y() * vector2.x();
    double angle = std::abs(std::atan2(cross, dot) * 180.0 / M_PI);
    if(angle > 90.0){
        angle = 180.0 - angle;
    }
    return angle <= 10.0;
}

CircleSet orderBigMarkers(
    const CircleSet& bigMarkers,
    const MarkerDistanceInfo& distanceInfo,
    int p3Index
){
    const int near1 = distanceInfo.nearest.first;
    const int near2 = distanceInfo.nearest.second;
    const int far1 = distanceInfo.farthest.first;
    const int far2 = distanceInfo.farthest.second;

    const cv::Point2d nearP1 = bigMarkers[near1].center;
    const cv::Point2d nearP2 = bigMarkers[near2].center;
    const cv::Point2d farP1 = bigMarkers[far1].center;
    const cv::Point2d farP2 = bigMarkers[far2].center;
    const cv::Point2d p3 = bigMarkers[p3Index].center;

    const cv::Point2d farVector = farP2 - farP1;
    const Eigen::Vector2d farDirection(farVector.x, farVector.y);
    const Eigen::Vector2d near1ToP3(p3.x - nearP1.x, p3.y - nearP1.y);
    const Eigen::Vector2d near2ToP3(p3.x - nearP2.x, p3.y - nearP2.y);

    const double angle1 = std::atan2(
        near1ToP3.x() * farDirection.y() - near1ToP3.y() * farDirection.x(),
        near1ToP3.dot(farDirection)
    ) * 180.0 / CV_PI;

    const double angle2 = std::atan2(
        near2ToP3.x() * farDirection.y() - near2ToP3.y() * farDirection.x(),
        near2ToP3.dot(farDirection)
    ) * 180.0 / CV_PI;

    int p0 = angle1 > angle2 ? near1 : near2;
    int p1 = angle1 > angle2 ? near2 : near1;

    const double distanceToFar1 = std::pow(bigMarkers[p0].center.x - farP1.x, 2) +
                                  std::pow(bigMarkers[p0].center.y - farP1.y, 2);
    const double distanceToFar2 = std::pow(bigMarkers[p0].center.x - farP2.x, 2) +
                                  std::pow(bigMarkers[p0].center.y - farP2.y, 2);

    const int p2 = distanceToFar1 > distanceToFar2 ? far1 : far2;
    const int p4 = distanceToFar1 > distanceToFar2 ? far2 : far1;

    return {bigMarkers[p0], bigMarkers[p1], bigMarkers[p2], bigMarkers[p3Index], bigMarkers[p4]};
}

}  // namespace

CircleSets sortMarkerCenters(const CircleSets& unsortedCenters){
    CircleSets sortedMarkerCenters;
    sortedMarkerCenters.reserve(unsortedCenters.size());

    for(const CircleSet& imageCircles : unsortedCenters){
        CircleSet bigMarkers = getBigMarkers(imageCircles);
        if(bigMarkers.size() < 5){
            sortedMarkerCenters.push_back({});
            continue;
        }

        const MarkerDistanceInfo distanceInfo = findNearestAndFarthestMarkers(bigMarkers);
        const int p3Index = findRemainingMarkerIndex(distanceInfo);
        if(p3Index == -1 || !isMarkerPairParallel(bigMarkers, distanceInfo)){
            sortedMarkerCenters.push_back({});
            continue;
        }

        sortedMarkerCenters.push_back(orderBigMarkers(bigMarkers, distanceInfo, p3Index));
    }

    return sortedMarkerCenters;
}

std::vector<Eigen::Matrix3d> findHomography(const CircleSets& sortedMarkerCenters){
    std::vector<Eigen::Matrix3d> homographies;
    homographies.reserve(sortedMarkerCenters.size());

    constexpr double centerDistance = 150.0;
    const std::vector<cv::Point2d> idealCenters = {
        {1 * centerDistance, 2 * centerDistance},
        {0 * centerDistance, 2 * centerDistance},
        {-3 * centerDistance, 0 * centerDistance},
        {0 * centerDistance, -2 * centerDistance},
        {3 * centerDistance, 0 * centerDistance},
    };

    for(const CircleSet& imageMarkers : sortedMarkerCenters){
        Eigen::MatrixXd a(10, 9);
        a.setZero();

        for(int circleIndex = 0; circleIndex < 5; ++circleIndex){
            const double x = idealCenters[circleIndex].x;
            const double y = idealCenters[circleIndex].y;
            const double u = imageMarkers[circleIndex].center.x;
            const double v = imageMarkers[circleIndex].center.y;

            a(circleIndex * 2, 0) = x;
            a(circleIndex * 2, 1) = y;
            a(circleIndex * 2, 2) = 1.0;
            a(circleIndex * 2, 6) = -u * x;
            a(circleIndex * 2, 7) = -u * y;
            a(circleIndex * 2, 8) = -u;

            a(circleIndex * 2 + 1, 3) = x;
            a(circleIndex * 2 + 1, 4) = y;
            a(circleIndex * 2 + 1, 5) = 1.0;
            a(circleIndex * 2 + 1, 6) = -v * x;
            a(circleIndex * 2 + 1, 7) = -v * y;
            a(circleIndex * 2 + 1, 8) = -v;
        }

        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
        const Eigen::VectorXd h = svd.matrixV().col(8);

        Eigen::Matrix3d homography;
        homography << h(0), h(1), h(2),
                      h(3), h(4), h(5),
                      h(6), h(7), h(8);

        if(std::abs(homography(2, 2)) > 1e-12){
            homography /= homography(2, 2);
        }

        homographies.push_back(homography);
    }

    return homographies;
}

CircleSets sortBoardCirclesByHomography(
    const std::vector<Eigen::Matrix3d>& homographies,
    const CircleSets& unsortedCircleCenters
){
    if(homographies.size() != unsortedCircleCenters.size()){
        return {};
    }

    CircleSets sortedBoardCircles;
    sortedBoardCircles.reserve(unsortedCircleCenters.size());

    for(size_t imageIndex = 0; imageIndex < unsortedCircleCenters.size(); ++imageIndex){
        const Eigen::Matrix3d inverseHomography = homographies[imageIndex].inverse();
        CircleSet imageCircles;
        imageCircles.reserve(unsortedCircleCenters[imageIndex].size());

        for(const Circle& inputCircle : unsortedCircleCenters[imageIndex]){
            Circle circle = inputCircle;
            const Eigen::Vector3d imagePoint(circle.center.x, circle.center.y, 1.0);
            const Eigen::Vector3d boardPoint = inverseHomography * imagePoint;
            if(std::abs(boardPoint(2)) < 1e-12){
                continue;
            }

            circle.board_col = boardPoint(0) / boardPoint(2);
            circle.board_row = boardPoint(1) / boardPoint(2);
            imageCircles.push_back(circle);
        }

        constexpr double rowTolerance = 7.5;
        std::sort(
            imageCircles.begin(),
            imageCircles.end(),
            [](const Circle& left, const Circle& right){
                if(std::abs(left.board_row - right.board_row) > rowTolerance){
                    return left.board_row < right.board_row;
                }
                return left.board_col < right.board_col;
            }
        );

        sortedBoardCircles.push_back(imageCircles);
    }

    return sortedBoardCircles;
}

}  // namespace camcalib::solver
