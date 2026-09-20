#pragma once

#include "core/CalibrationData.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace camcalib {

/** @brief 圆点标定板的几何配置。 */
struct BoardConfig {
    int rows = 0;              ///< 圆点行数。
    int cols = 0;              ///< 圆点列数。
    double spacingMm = 0.0;    ///< 相邻圆心距离，单位为毫米。
};

/** @brief 圆点轮廓检测、筛选和排序配置。 */
struct DetectorConfig {
    int minContourPoints = 100;                   ///< 轮廓允许的最少离散点数。
    double minContourArea = 100.0;                ///< 轮廓最小面积，单位为像素平方。
    double maxContourArea = 100000.0;             ///< 轮廓最大面积，单位为像素平方。
    double maxAxisRatio = 1.5;                    ///< 外接矩形长短轴比例上限。
    int markerCount = 5;                          ///< 用于定位标定板的 Marker 数量。
    double markerSpacing = 150.0;                 ///< Marker 模型间距。
    double rowTolerance = 7.5;                    ///< 圆点按行排序时的容差。
    bool enableSubpixel = true;                   ///< 是否启用亚像素边缘细化。
    bool blackCirclesOnWhiteBackground = true;    ///< 是否使用白底黑圆标定板。
};

/** @brief 伪相机法投影仪标定配置。 */
struct ProjectorConfig {
    bool enabled = true;  ///< 是否执行投影仪标定。
    bool useCuda = false;  ///< 是否使用 CUDA 计算绝对相位。
    std::string method = "pseudo_camera";  ///< 投影仪标定方法名称。
    std::string calibrationDataDirectory;  ///< 投影仪多位姿相位图根目录。
    std::array<float, 3> phaseFrequencies = {64.0f, 16.0f, 4.0f};  ///< 从高到低的三频条纹频率。
    int width = 1920;       ///< 投影仪有效分辨率宽度。
    int height = 1080;      ///< 投影仪有效分辨率高度。
    int minValidViews = 3;  ///< 投影仪标定要求的最少有效位姿数。
};

/** @brief 调试图像与交互显示配置。 */
struct DebugConfig {
    bool enabled = false;                            ///< 是否启用交互调试功能。
    bool saveImages = true;                          ///< 是否保存调试图像。
    bool showWindows = false;                        ///< 是否显示 OpenCV 调试窗口。
    std::string outputDirectory = "debug_output";    ///< 调试输出根目录。
};

/** @brief 相机标定图像数据集配置。 */
struct DatasetConfig {
    std::string imageDirectory;                 ///< 相机标定图像目录。
    std::vector<std::string> imageExtensions;   ///< 允许读取的图像扩展名。
    bool readGrayscale = true;                  ///< 是否以灰度模式读取图像。
};

/** @brief 运行日志配置。 */
struct LoggingConfig {
    bool enabled = true;                              ///< 是否写入日志。
    std::string outputFile = "debug_output/run.log";  ///< 日志文件路径。
};

/** @brief 相机和投影仪标定 Pipeline 的完整配置。 */
struct CalibrationPipelineConfig {
    DatasetConfig dataset;       ///< 相机数据集配置。
    BoardConfig board;           ///< 标定板配置。
    DetectorConfig detector;     ///< 圆点检测配置。
    ProjectorConfig projector;   ///< 投影仪标定配置。
    DebugConfig debug;           ///< 调试输出配置。
    LoggingConfig logging;       ///< 日志配置。
};

/** @brief 一幅已加载图像及其来源路径。 */
struct DatasetImage {
    std::string path;  ///< 图像文件路径或位姿名称。
    cv::Mat image;     ///< 图像像素数据。
};

/** @brief 尺寸一致的标定图像集合。 */
struct CalibrationDataset {
    std::vector<DatasetImage> images;  ///< 数据集中的图像。
    cv::Size imageSize;                ///< 数据集统一图像尺寸。

    /** @brief 判断数据集是否不包含图像。 */
    bool empty() const { return images.empty(); }

    /** @brief 返回数据集图像数量。 */
    size_t size() const { return images.size(); }
};

/** @brief 单个位姿下的世界点和图像点观测。 */
struct ViewObservation {
    std::string imagePath;                   ///< 当前位姿的图像路径或名称。
    cv::Size imageSize;                      ///< 当前位姿图像尺寸。
    std::vector<cv::Point2d> imagePoints;    ///< 相机或投影仪像素坐标。
    std::vector<cv::Point3f> objectPoints;   ///< 与像素点对应的世界坐标。

    bool valid = false;              ///< 当前观测是否可用于标定。
    std::string failureReason;       ///< 无效观测的原因。

    double detectionScore = 0.0;     ///< 有效检测点占期望点数的比例。
    double reprojectionRmse = 0.0;   ///< 当前位姿重投影均方根误差。
};

/** @brief 圆点检测各阶段的结果集合。 */
struct DetectionResult {
    std::vector<ViewObservation> views;  ///< 每个位姿的最终标定观测。
    std::vector<std::vector<std::vector<cv::Point>>> pixelEdges;  ///< 每幅图像的整像素轮廓。
    std::vector<std::vector<std::vector<cv::Point2d>>> subPixelEdges;  ///< 每幅图像的亚像素轮廓。
    std::vector<std::vector<Circle>> fittedCircles;  ///< 所有候选轮廓的拟合圆。
    std::vector<std::vector<Circle>> sortedMarkerCircles;  ///< 排序后的定位 Marker。
    std::vector<Eigen::Matrix3d> homographies;  ///< 标定板平面到图像平面的单应矩阵。
    std::vector<std::vector<Circle>> sortedBoardCircles;  ///< 排序后的全部标定板圆点。
};

/** @brief 针孔设备的内参、畸变、外参和误差结果。 */
struct CalibrationResult {
    cv::Mat cameraMatrix;                  ///< 相机或投影仪的 3x3 内参矩阵。
    cv::Mat distCoeffs;                    ///< 镜头畸变系数。
    std::vector<cv::Mat> rotationVectors;  ///< 每个位姿的旋转向量。
    std::vector<cv::Mat> translationVectors;  ///< 每个位姿的平移向量。

    double globalRmse = 0.0;  ///< 全局重投影均方根误差，单位为像素。
    bool converged = false;   ///< 标定是否成功生成有效参数。
    std::string solverName;   ///< 标定求解器名称。
};

/** @brief 相机坐标系到投影仪坐标系的联合标定结果。 */
struct CameraProjectorCalibrationResult {
    cv::Mat rotation;           ///< 相机坐标系到投影仪坐标系的 3x3 旋转矩阵。
    cv::Mat translation;        ///< 相机坐标系到投影仪坐标系的 3x1 平移向量。
    cv::Mat essentialMatrix;    ///< 相机与投影仪之间的本质矩阵。
    cv::Mat fundamentalMatrix;  ///< 相机与投影仪之间的基础矩阵。
    double globalRmse = 0.0;    ///< 联合标定全局重投影 RMSE。
    bool converged = false;     ///< 是否成功生成有效相对外参。
};

/** @brief 单个位姿的重投影误差统计。 */
struct ViewError {
    size_t viewIndex = 0;       ///< 位姿在原始观测数组中的索引。
    double rmse = 0.0;          ///< 当前位姿均方根误差，单位为像素。
    double maxError = 0.0;      ///< 当前位姿最大点误差，单位为像素。
    cv::Point2d meanError;      ///< 当前位姿平均二维误差向量。
};

/** @brief 整体标定质量评价报告。 */
struct EvaluationReport {
    double globalRmse = 0.0;              ///< 标定器返回的全局均方根误差。
    double meanViewRmse = 0.0;            ///< 所有有效位姿 RMSE 的平均值。
    double maxViewRmse = 0.0;             ///< 最大位姿 RMSE。
    std::vector<ViewError> views;          ///< 各有效位姿误差。
    std::vector<size_t> suspectedOutliers; ///< 疑似异常位姿索引。
};

}  // namespace camcalib
