// 相位解算Cuda 版本实现


#pragma once

#include "imageProcess/ThreeFrequencyFourStepPhase.h"

namespace camcalib {

class ThreeFrequencyFourStepPhaseCuda
{
public:
    /**
     * @brief 使用 CUDA 根据四步相移图像计算包裹相位
     *
     * 输入四张 CV_32FC1 图像：
     * image0, image1, image2, image3
     *
     * 计算公式：
     * phase = atan2(image3 - image1, image0 - image2)
     *
     * @return CV_32FC1 包裹相位图
     */
    static cv::Mat calculateWrappedPhase(
        const cv::Mat& image0,
        const cv::Mat& image1,
        const cv::Mat& image2,
        const cv::Mat& image3
    );

     /** @brief 计算两幅包裹相位的模 2π 差相位。
     *  @param higherFrequencyPhase 较高频率包裹相位。
     *  @param lowerFrequencyPhase 较低频率包裹相位。
     *  @return 范围为 [0, 2π) 的合成相位。
     */
    static cv::Mat calculateSyntheticPhaseCuda(
        const cv::Mat& higherFrequencyPhase,
        const cv::Mat& lowerFrequencyPhase
    );



   /** @brief 完整执行三频四步绝对相位解算。
     *  @param images 按频率从高到低排列的12幅 CV_32FC1 图像。
     *  @param frequencies 严格从高到低排列的三个频率。
     *  @return 包裹相位、合成相位和绝对相位。
     */
    static ThreeFrequencyPhaseResult solveCuda(
        const std::vector<cv::Mat>& images,
        const std::array<float, 3>& frequencies
    );

       /** @brief 使用两级合成相位展开最高频率相位。
     *  @param syntheticPhase123 三频合成相位。
     *  @param syntheticPhase23 第二、第三频率合成相位。
     *  @param highestWrappedPhase 最高频率包裹相位。
     *  @param frequency1 最高频率。
     *  @param frequency2 中间频率。
     *  @param frequency3 最低频率。
     *  @return 展开后的最高频率绝对相位。
     */
    static cv::Mat unwrapHighestFrequencyCuda(
        const cv::Mat& syntheticPhase123,
        const cv::Mat& syntheticPhase23,
        const cv::Mat& highestWrappedPhase,
        float frequency1,
        float frequency2,
        float frequency3
    );

};



} // namespace camcalib
