#include "imageProcess/ThreeFrequencyFourStepPhase.h"

#include <cmath>

namespace camcalib {
namespace {

constexpr float kTwoPi = 6.28318530717958647692f;

void checkFloatPhaseImage(const cv::Mat& image){
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_32FC1);
}

void checkSameSizeAndType(const cv::Mat& first, const cv::Mat& second){
    checkFloatPhaseImage(first);
    checkFloatPhaseImage(second);
    CV_Assert(first.size() == second.size());
}

}  // namespace

cv::Mat ThreeFrequencyFourStepPhase::calculateWrappedPhase(
    const std::vector<cv::Mat>& images,
    int frequencyIndex
){
    CV_Assert(images.size() == 12);
    CV_Assert(frequencyIndex >= 0 && frequencyIndex < 3);

    const int offset = frequencyIndex * 4;
    for(int index = 0; index < 4; ++index){
        checkFloatPhaseImage(images[offset + index]);
        CV_Assert(images[offset + index].size() == images[offset].size());
    }

    const cv::Mat numerator = images[offset + 3] - images[offset + 1];
    const cv::Mat denominator = images[offset] - images[offset + 2];
    cv::Mat wrappedPhase(images[offset].size(), CV_32FC1);

    for(int row = 0; row < wrappedPhase.rows; ++row){
        const float* numeratorPtr = numerator.ptr<float>(row);
        const float* denominatorPtr = denominator.ptr<float>(row);
        float* phasePtr = wrappedPhase.ptr<float>(row);
        for(int col = 0; col < wrappedPhase.cols; ++col){
            phasePtr[col] = std::atan2(numeratorPtr[col], denominatorPtr[col]);
        }
    }
    return wrappedPhase;
}

cv::Mat ThreeFrequencyFourStepPhase::calculateSyntheticPhase(
    const cv::Mat& higherFrequencyPhase,
    const cv::Mat& lowerFrequencyPhase
){
    checkSameSizeAndType(higherFrequencyPhase, lowerFrequencyPhase);
    cv::Mat syntheticPhase(higherFrequencyPhase.size(), CV_32FC1);

    for(int row = 0; row < syntheticPhase.rows; ++row){
        const float* highPtr = higherFrequencyPhase.ptr<float>(row);
        const float* lowPtr = lowerFrequencyPhase.ptr<float>(row);
        float* resultPtr = syntheticPhase.ptr<float>(row);
        for(int col = 0; col < syntheticPhase.cols; ++col){
            float phaseDifference = highPtr[col] - lowPtr[col];
            if(phaseDifference < 0.0f){
                phaseDifference += kTwoPi;
            }
            resultPtr[col] = phaseDifference;
        }
    }
    return syntheticPhase;
}

cv::Mat ThreeFrequencyFourStepPhase::unwrapHighestFrequency(
    const cv::Mat& syntheticPhase123,
    const cv::Mat& syntheticPhase23,
    const cv::Mat& highestWrappedPhase,
    float frequency1,
    float frequency2,
    float frequency3
){
    checkSameSizeAndType(syntheticPhase123, syntheticPhase23);
    checkSameSizeAndType(syntheticPhase123, highestWrappedPhase);
    CV_Assert(frequency1 > frequency2 && frequency2 > frequency3);

    //const float referenceLength = static_cast<float>(highestWrappedPhase.cols);
    const float period1 = 1.0 / frequency1;
    const float period2 = 1.0 / frequency2;
    const float period3 = 1.0 / frequency3;
    const float period12 = period1 * period2 / (period2 - period1);
    const float period23 = period2 * period3 / (period3 - period2);
    const float period123 = period12 * period23 / (period23 - period12);

    cv::Mat unwrappedPhase23(syntheticPhase123.size(), CV_32FC1);
    cv::Mat unwrappedPhase1(syntheticPhase123.size(), CV_32FC1);

    for(int row = 0; row < syntheticPhase123.rows; ++row){
        for(int col = 0; col < syntheticPhase123.cols; ++col){
            const float phase123 = syntheticPhase123.at<float>(row, col);
            const float phase23 = syntheticPhase23.at<float>(row, col);
            unwrappedPhase23.at<float>(row, col) =
                phase23 + kTwoPi * static_cast<float>(std::round(
                    (phase123 * period123 / period23 - phase23 ) / kTwoPi
                ));
        }
    }

    for(int row = 0; row < syntheticPhase123.rows; ++row){
        for(int col = 0; col < syntheticPhase123.cols; ++col){
            const float phase1 = highestWrappedPhase.at<float>(row, col);
            const float phase23 = unwrappedPhase23.at<float>(row, col);
            unwrappedPhase1.at<float>(row, col) =
                phase1 + kTwoPi * static_cast<float>(std::round(
                    (phase23 * period23 / period1 - phase1) / kTwoPi
                ));
        }
    }
    return unwrappedPhase1;
}

ThreeFrequencyPhaseResult ThreeFrequencyFourStepPhase::solve(
    const std::vector<cv::Mat>& images,
    const std::array<float, 3>& frequencies
){
    ThreeFrequencyPhaseResult result;
    for(int index = 0; index < 3; ++index){
        result.wrappedPhases[index] = calculateWrappedPhase(images, index);
    }

    const cv::Mat syntheticPhase12 = calculateSyntheticPhase(
        result.wrappedPhases[0],
        result.wrappedPhases[1]
    );
    result.syntheticPhase23 = calculateSyntheticPhase(
        result.wrappedPhases[1],
        result.wrappedPhases[2]
    );
    result.syntheticPhase123 = calculateSyntheticPhase(
        syntheticPhase12,
        result.syntheticPhase23
    );
    result.unwrappedPhase = unwrapHighestFrequency(
        result.syntheticPhase123,
        result.syntheticPhase23,
        result.wrappedPhases[0],
        frequencies[0],
        frequencies[1],
        frequencies[2]
    );
    return result;
}

}  // namespace camcalib
