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

    const int index = row * col + col;   //  像素再图像指针中的下标

    const float numerator = image3[index] - image0[index];
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


cv::Mat calculateSyntheticPhaseCuda(
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





ThreeFrequencyPhaseResult ThreeFrequencyFourStepPhaseCuda::solveCuda(
    const std::vector<cv::Mat>& images,
    const std::array<float, 3>& frequencies
){

    ThreeFrequencyPhaseResult result;
    for(int index = 0; index < 3; ++index){
        result.wrappedPhases[index] = calculateWrappedPhase(images[index], images[index+1], images[index+2], images[index+3]);
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

    std::cerr<<"cuda end"  << std::endl;

    return result;
}



}