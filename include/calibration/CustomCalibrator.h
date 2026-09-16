#pragma once

#include "core/CalibrationTypes.h"
#include <opencv2/calib3d.hpp>


namespace camcalib {

/** @brief 自定义平面标定算法的实验性实现。 */
class CustomCalibrator {
public:
    /** @brief 创建自定义标定器。
     *  @param boardConfig 标定板几何配置。
     */
    explicit CustomCalibrator(BoardConfig boardConfig);

    /** @brief 使用自定义流程估计标定参数。 */
    CalibrationResult calibrate(const CalibrationDataset& dataset, 
                                const DetectionResult& detetion) const;
                                
private:
    /** @brief 估计单个位姿的平面单应矩阵。 */
    Eigen::Matrix3d estimateHomography(
        const std::vector<cv::Point3f>& objectPoints,
        const std::vector<cv::Point2d>& imagePoints
    ) const;

    /** @brief 估计全部有效位姿的平面单应矩阵。 */
    std::vector<Eigen::Matrix3d> estimateAllPoseHomography(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2d>>& imagePoints
    ) const;

    /** @brief 构造 Zhang 标定约束向量 v_ij。 */
    Eigen::Matrix<double, 6, 1> makeV(
        const Eigen::Matrix3d& H,
        int i, int j 
    ) const;

    /** @brief 根据多幅单应矩阵估计内参初值。 */
    cv::Mat estimateIntrinsics(
        const std::vector<Eigen::Matrix3d>& homographies
    ) const;

    /** @brief 根据内参和单应矩阵估计各位姿外参。 */
    void estimateExtrinsics(
        const cv::Mat& cameraMatrix,
        const std::vector<cv::Mat>& homographies,
        std::vector<cv::Mat>& rotationVectors,
        std::vector<cv::Mat>& translationVectors
    ) const;

    /** @brief 初始化镜头畸变系数。 */
    cv::Mat initializeDistortion(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2d>>& imagePoints,
        const cv::Mat& cameraMatrix,
        const std::vector<cv::Mat>& rotationVectors,
        const std::vector<cv::Mat>& translationVectors
    ) const;

    /** @brief 联合优化内参、畸变和各位姿外参。 */
    void bundleAdjustment(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2d>>& imagePoints,
        CalibrationResult& result
    ) const;

    /** @brief 计算全部有效观测的重投影 RMSE。 */
    double calculateRmse(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2d>>& imagePoints,
        const CalibrationResult& result
    ) const;

    /** @brief 归一化一组二维坐标并返回归一化矩阵。 */
    std::pair<std::vector<cv::Point2d>, Eigen::Matrix3d> normalizeObjectCoordinates(
        const std::vector<cv::Point2d>& coordinates
    ) const;

    /** @brief 将平面三维世界点转换为二维坐标。 */
    std::vector<cv::Point2d> point3f2point2d(
        const std::vector<cv::Point3f>& worldPoints
    ) const;

    BoardConfig boardConfig_;  ///< 标定板几何配置。
};

}  // namespace camcalib
