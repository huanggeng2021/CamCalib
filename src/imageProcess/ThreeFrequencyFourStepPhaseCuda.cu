#include "imageProcess/ThreeFrequencyFourStepPhaseCuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>   
#include <iostream>

namespace camcalib{

void checkCudaError(cudaError_t error, const char* message){
    if(error != cudaSuccess){

        throw std::runtime_error(
            std::string(message) + ": "+ cudaGetErrorString(error)
        );
    }

}

// 计算包裹相位核函数
// static 修饰符的作用是 该核函数只在当前cu文件中有效
static __global__ void calculateWrappedPhaseKernel(
    const float* image0,
    const float* image1,
    const float* image2,
    const float* image3,
    int rows,
    int cols,
    float* wrappedPhase
){

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if( row >= rows || col >= cols){
        return;
    }

    const int index = row * cols + col;   //  像素再图像指针中的下标

    const float numerator = image3[index] - image1[index];
    const float denominator = image0[index] - image2[index];

    wrappedPhase[index] = atan2f(numerator, denominator); 
}

static __global__ void calculateSyntheticPhaseKernel(
    const float* higherFrequencyPhase,
    const float* lowerFrequencyPhase,
    int rows,
    int cols,
    float* syntheticPhase
){

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col >= cols || row >= rows){
        return;
    }

    const int index = row * cols + col;

    float phaseDifference = higherFrequencyPhase[index] - lowerFrequencyPhase[index];

    if(phaseDifference < 0.0f){    // 会有分支分化

        phaseDifference += 6.28318530717958647692f;
    }

    syntheticPhase[index] = phaseDifference;

}


static __global__ void unwrapPhaseKernel(
    const float* coarseUnwrappedPhase,
    const float* wrappedPhase,
    float coarsePeriod,
    float targetPeriod,
    int cols,
    int rows,
    float* unwrappedPhase
) {

    const float kTwoPi = 6.28318530717958647692f;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col >= cols || row >= rows){
        return;
    }

    const int index = row * cols + col;

    unwrappedPhase[index] = wrappedPhase[index] + kTwoPi * 
        roundf((coarseUnwrappedPhase[index] * coarsePeriod / targetPeriod - wrappedPhase[index] ) / kTwoPi);

}


cv::Mat ThreeFrequencyFourStepPhaseCuda::calculateWrappedPhase(
    const cv::Mat& image0,
    const cv::Mat& image1,
    const cv::Mat& image2,
    const cv::Mat& image3)
{

    // -------------------------
    // 1. 输入检查
    // -------------------------

    CV_Assert(!image0.empty());
    CV_Assert(!image1.empty());
    CV_Assert(!image2.empty());
    CV_Assert(!image3.empty());

    CV_Assert(image0.type() == CV_32FC1);
    CV_Assert(image1.type() == CV_32FC1);
    CV_Assert(image2.type() == CV_32FC1);
    CV_Assert(image3.type() == CV_32FC1);

    CV_Assert(image0.size() == image1.size());
    CV_Assert(image0.size() == image2.size());
    CV_Assert(image0.size() == image3.size());

    const int rows = image0.rows;
    const int cols = image0.cols;

    const size_t pixelCount = static_cast<size_t>(rows) * cols;
    const size_t byteCount = pixelCount * sizeof(float);   //  按照字节大小来开辟内存


    // 分配GPU内存  device 端

    float* deviceImage0 = nullptr;
    float* deviceImage1 = nullptr;
    float* deviceImage2 = nullptr;
    float* deviceImage3 = nullptr;

    float* deviceWrappedPhase = nullptr;

    checkCudaError(cudaMalloc(&deviceImage0, byteCount), "cudaMalloc deviceImage0 failed");
    checkCudaError(cudaMalloc(&deviceImage1, byteCount), "cudaMalloc deviceImage1 failed");
    checkCudaError(cudaMalloc(&deviceImage2, byteCount), "cudaMalloc deviceImage2 failed");
    checkCudaError(cudaMalloc(&deviceImage3, byteCount), "cudaMalloc deviceImage3 failed");
    checkCudaError(cudaMalloc(&deviceWrappedPhase, byteCount), "cudaMalloc deviceWrappedPhase failed");

    // copy data form cpu to gpu
    checkCudaError(cudaMemcpy(
        deviceImage0, image0.ptr<float>(), byteCount, cudaMemcpyHostToDevice),
        "copy image0 failed"
    );
    checkCudaError(cudaMemcpy(
        deviceImage1, image1.ptr<float>(), byteCount, cudaMemcpyHostToDevice),
        "copy image1 failed"
    );
    checkCudaError(cudaMemcpy(
        deviceImage2, image2.ptr<float>(), byteCount, cudaMemcpyHostToDevice),
        "copy image2 failed"
    );
     checkCudaError(cudaMemcpy(
        deviceImage3, image3.ptr<float>(), byteCount, cudaMemcpyHostToDevice),
        "copy image3 failed"
    );

    // cuda 线程配置
    const dim3 block(16, 16);

    const dim3 grid(
        (cols + block.x -1) / block.x,
        (rows + block.y -1) / block.y
    );

    // 启动核函数
    calculateWrappedPhaseKernel<<<grid, block>>>(
        deviceImage0, deviceImage1,
        deviceImage2, deviceImage3,
        rows, cols, 
        deviceWrappedPhase
    );

    // 等待同步，方便调试
    checkCudaError(
        cudaDeviceSynchronize(),
        "calculateWrappedPhaseKernel execution failed"
    );

    // GPU - CPU
    cv::Mat wrappedPhase(rows, cols, CV_32F);

    checkCudaError(
        cudaMemcpy(wrappedPhase.ptr<float>(), deviceWrappedPhase, byteCount, cudaMemcpyDeviceToHost),
        "copy wrapped phase failed"
    );



    // 释放内存
    cudaFree(deviceImage0);
    cudaFree(deviceImage1);
    cudaFree(deviceImage2);
    cudaFree(deviceImage3);

    cudaFree(deviceWrappedPhase);

    return wrappedPhase;

}


cv::Mat ThreeFrequencyFourStepPhaseCuda::calculateSyntheticPhaseCuda(
    const cv::Mat& higherFrequencyPhase,
    const cv::Mat& lowerFrequencyPhase
    ){

    CV_Assert(!higherFrequencyPhase.empty());
    CV_Assert(!lowerFrequencyPhase.empty());

    CV_Assert(higherFrequencyPhase.type() == CV_32FC1);
    CV_Assert(lowerFrequencyPhase.type() == CV_32FC1);

    CV_Assert(higherFrequencyPhase.size() == lowerFrequencyPhase.size());

    CV_Assert(higherFrequencyPhase.isContinuous());
    CV_Assert(lowerFrequencyPhase.isContinuous());

    const int cols = higherFrequencyPhase.cols;
    const int rows = higherFrequencyPhase.rows;
    
    float* deviceSyntheticPhase = nullptr;
    float* deviceHigherFrequencyPhase = nullptr;
    float* deviceLowerFrequencyPhase = nullptr;
    
    // 计算大小
    const size_t byteSize = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    
    // 开辟device端内存

    checkCudaError(cudaMalloc(&deviceSyntheticPhase, byteSize), "wrapPhase malloc failed");
    checkCudaError(cudaMalloc(&deviceHigherFrequencyPhase, byteSize), "deviceHigherFrequencyPhase failed");
    checkCudaError(cudaMalloc(&deviceLowerFrequencyPhase, byteSize), "deviceLowerFrequencyPhase failed");

    checkCudaError(cudaMemcpy(deviceHigherFrequencyPhase, higherFrequencyPhase.ptr<float>(), byteSize, cudaMemcpyHostToDevice),
                    " copy deviceHigherFrequencyPhase failed");
    checkCudaError(cudaMemcpy(deviceLowerFrequencyPhase, lowerFrequencyPhase.ptr<float>(), byteSize, cudaMemcpyHostToDevice),
                    "copy deviceLowerFrequencyPhase failed");

    // 配置cuda参数

    dim3 block(16, 16);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    calculateSyntheticPhaseKernel<<<grid, block>>>(
        deviceHigherFrequencyPhase, deviceLowerFrequencyPhase,
        rows, cols,
        deviceSyntheticPhase
        );
    checkCudaError(cudaGetLastError(), "calculateSyntheticPhaseKernel launch failed");

    // 等待同步
    checkCudaError(
        cudaDeviceSynchronize(),
        "calculateWrappedPhaseKernel execution failed"
    );


    // gpu -> cpu
    cv::Mat syntheticPhaseHost(rows, cols, CV_32F);
    checkCudaError(cudaMemcpy(syntheticPhaseHost.ptr<float>(), deviceSyntheticPhase, byteSize, cudaMemcpyDeviceToHost),
     "unWrapPhaseHost copy failed"); 

    // 释放内存
    cudaFree(deviceSyntheticPhase);
    cudaFree(deviceHigherFrequencyPhase);
    cudaFree(deviceLowerFrequencyPhase);

    return syntheticPhaseHost;
}

cv::Mat ThreeFrequencyFourStepPhaseCuda::unwrapHighestFrequencyCuda(
    const cv::Mat& syntheticPhase123,
    const cv::Mat& syntheticPhase23,
    const cv::Mat& highestWrappedPhase,
    float frequency1,
    float frequency2,
    float frequency3
){

    const float period1 = 1.0 / frequency1;
    const float period2 = 1.0 / frequency2;
    const float period3 = 1.0 / frequency3;
    const float period12 = period1 * period2 / (period2 - period1);
    const float period23 = period2 * period3 / (period3 - period2);
    const float period123 = period12 * period23 / (period23 - period12);

  

    // 开辟device内存
    const int rows = syntheticPhase123.rows;
    const int cols = syntheticPhase123.cols;
    // 计算字节数
    size_t byteSize = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);

    float* syntheticPhase123Device = nullptr;
    float* syntheticPhase23Device = nullptr;
    float* highestWrappedPhaseDevice = nullptr;

    float* unwrappedPhase23Device = nullptr;
    float* unwrappedPhase1Device = nullptr;

    checkCudaError(cudaMalloc(&syntheticPhase123Device, byteSize), 
        "syntheticPhase123Device malloc failed");
    checkCudaError(cudaMalloc(&syntheticPhase23Device, byteSize), 
        "syntheticPhase23Device malloc failed");
    checkCudaError(cudaMalloc(&highestWrappedPhaseDevice, byteSize), 
        "highestWrappedPhaseDevice malloc failed");

    checkCudaError(cudaMalloc(&unwrappedPhase23Device, byteSize), 
        "unwrappedPhase23Device malloc failed");
    checkCudaError(cudaMalloc(&unwrappedPhase1Device, byteSize), 
        "unwrappedPhase1Device malloc failed");

    // 拷贝 host to device
    
    checkCudaError(cudaMemcpy(syntheticPhase123Device, syntheticPhase123.ptr<float>(), byteSize, cudaMemcpyHostToDevice),
        "syntheticPhase123Device cpoy failed");
    checkCudaError(cudaMemcpy(syntheticPhase23Device, syntheticPhase23.ptr<float>(), byteSize, cudaMemcpyHostToDevice),
        "syntheticPhase23Device cpoy failed");
    checkCudaError(cudaMemcpy(highestWrappedPhaseDevice, highestWrappedPhase.ptr<float>(), byteSize, cudaMemcpyHostToDevice),
        "highestWrappedPhaseDevice cpoy failed");
    
    // 配置核函数
    dim3 block(16, 16);

    dim3 grid(
        (cols + block.x - 1) / block.x ,
        (rows + block.y - 1) /  block.y
    );

    unwrapPhaseKernel<<<grid, block>>>(
        syntheticPhase123Device,
        syntheticPhase23Device,
        period123, 
        period23,
        cols, rows,
        unwrappedPhase23Device
    );

    unwrapPhaseKernel<<<grid, block>>>(
        unwrappedPhase23Device,
        highestWrappedPhaseDevice,
        period23, 
        period1,
        cols, rows,
        unwrappedPhase1Device
    );

    // 等待同步
    checkCudaError(
        cudaDeviceSynchronize(),
        "calculateWrappedPhaseKernel execution failed"
    );

    // device to host 
    cv::Mat unwrappedPhase1(syntheticPhase123.size(), CV_32FC1);  // 最终绝对相位
    checkCudaError(cudaMemcpy(unwrappedPhase1.ptr<float>(), unwrappedPhase1Device, byteSize, cudaMemcpyDeviceToHost), 
        "unwrapPhaseKernel execution failed");


    // 释放内存
    cudaFree(syntheticPhase123Device);
    cudaFree(syntheticPhase23Device);
    cudaFree(highestWrappedPhaseDevice);

    cudaFree(unwrappedPhase23Device);
    cudaFree(unwrappedPhase1Device);

    return unwrappedPhase1;
}


ThreeFrequencyPhaseResult ThreeFrequencyFourStepPhaseCuda::solveCuda(
    const std::vector<cv::Mat>& images,
    const std::array<float, 3>& frequencies
){

    ThreeFrequencyPhaseResult result;
    for (int index = 0; index < 3; ++index) {
        const int offset = index * 4;

        result.wrappedPhases[index] = calculateWrappedPhase(
            images[offset],
            images[offset + 1],
            images[offset + 2],
            images[offset + 3]
        );
    }

    // 计算合成相位

    const cv::Mat syntheticPhase12 = calculateSyntheticPhaseCuda(
        result.wrappedPhases[0],
        result.wrappedPhases[1]
    );
    result.syntheticPhase23 = calculateSyntheticPhaseCuda(
        result.wrappedPhases[1],
        result.wrappedPhases[2]
    );
    result.syntheticPhase123 = calculateSyntheticPhaseCuda(
        syntheticPhase12,
        result.syntheticPhase23
    );

    // 展开最高频率相位
    result.unwrappedPhase = unwrapHighestFrequencyCuda(
        result.syntheticPhase123,
        result.syntheticPhase23,
        result.wrappedPhases[0],
        frequencies[0],
        frequencies[1],
        frequencies[2]
    );

    std::cerr<<"cuda end"  << std::endl;

    return result;
}



}