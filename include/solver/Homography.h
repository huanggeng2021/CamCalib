#pragma once

#include "core/CalibrationData.h"
#include <Eigen/Dense>
#include <vector>

namespace camcalib::solver {

CircleSets sortMarkerCenters(const CircleSets& unsortedCenters);

std::vector<Eigen::Matrix3d> findHomography(const CircleSets& sortedMarkerCenters);

CircleSets sortBoardCirclesByHomography(
    const std::vector<Eigen::Matrix3d>& homographies,
    const CircleSets& unsortedCircleCenters
);

}  // namespace camcalib::solver
