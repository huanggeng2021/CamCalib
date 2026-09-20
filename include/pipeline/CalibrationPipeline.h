#pragma once

#include "core/CalibrationTypes.h"
#include "dataset/ProjectorDatasetLoader.h"
#include "calibration/CustomCalibrator.h"

#include <array>
#include <string>
#include <vector>

namespace camcalib {

/** @brief 编排相机检测、相机标定和伪相机法投影仪标定流程。 */
class CalibrationPipeline {
public:
    /** @brief 执行相机数据加载、圆点检测、标定和评价。
     *  @param configPath YAML 配置文件路径。
     *  @param cameraDetection 输出供投影仪流程复用的相机圆心和世界点。
     *  @param cameraCalibration 输出相机内参、畸变和各位姿外参。
     *  @return 相机标定是否成功。
     */
    bool runCameraCalibration(
        const std::string& configPath,
        DetectionResult& cameraDetection,
        CalibrationResult& cameraCalibration
    ) const;

    /** @brief 执行投影仪数据加载、相位解算、点匹配、标定和评价。
     *  @param configPath YAML 配置文件路径。
     *  @param cameraViews 相机流程生成的各位姿圆心和世界点。
     *  @param projectorPoses 输出包含绝对相位和投影仪特征点的位姿数据。
     *  @param projectorCalibration 输出投影仪内参、畸变和各位姿外参。
     *  @return 投影仪标定是否成功。
     */
    bool runProjectorCalibration(
        const std::string& configPath,
        const std::vector<ViewObservation>& cameraViews,
        std::vector<ProjectorPoseData>& projectorPoses,
        CalibrationResult& projectorCalibration
    ) const;

    /** @brief 固定单目标定内参，计算相机到投影仪的相对外参。
     *  @param configPath YAML 配置文件路径。
     *  @param cameraViews 相机流程生成的各位姿圆心和世界点。
     *  @param projectorPoses 投影仪流程生成的对应像素坐标。
     *  @param cameraCalibration 相机单目标定结果。
     *  @param projectorCalibration 投影仪单目标定结果。
     *  @return 联合标定是否成功。
     */
    bool runCameraProjectorCalibration(
        const std::string& configPath,
        const std::vector<ViewObservation>& cameraViews,
        const std::vector<ProjectorPoseData>& projectorPoses,
        const CalibrationResult& cameraCalibration,
        const CalibrationResult& projectorCalibration
    ) const;

private:
    /** @brief 组装投影仪标定器和评价器需要的观测结果。 */
    DetectionResult buildProjectorDetection(
        const std::vector<ViewObservation>& cameraViews,
        const std::vector<ProjectorPoseData>& poses,
        cv::Size projectorSize
    ) const;

    /** @brief 为所有投影仪位姿计算 X/Y 绝对相位。 */
    bool solveProjectorPhases(
        std::vector<ProjectorPoseData>& poses,
        const std::array<float, 3>& frequencies,
        int minValidViews,
        bool useCuda
    ) const;

    /** @brief 执行相机或投影仪共用的标定和重投影评价。 */
    CalibrationResult calibrateAndEvaluate(
        const CalibrationDataset& dataset,
        DetectionResult& detection,
        const BoardConfig& board,
        EvaluationReport& evaluation
    ) const;
};

}  // namespace camcalib
