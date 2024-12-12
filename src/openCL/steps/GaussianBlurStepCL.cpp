#include "GaussianBlurStepCL.h"
#include <cmath>
#include <stdexcept>

GaussianBlurStepCL::GaussianBlurStepCL(OpenCLManager& manager, int kernelSize,
                                       float sigma)
    : openclManager(manager)
    , kernelSize(kernelSize)
    , sigma(sigma)
{
    openclManager.loadKernel("gaussian_blur",
                             "src/opencl/kernels/gaussian_blur.cl");
    kernel = openclManager.getKernel("gaussian_blur");
    precomputeKernel();
}

void GaussianBlurStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Create OpenCL buffers
    cl::Image2D inputBuffer = openclManager.createImage2DFromMat(
        inputImage, 0 /* Assume GPU is pre-selected */);
    cl::Image2D outputBuffer(openclManager.getContext(), CL_MEM_WRITE_ONLY,
                             cl::ImageFormat(CL_R, CL_FLOAT), width, height);

    cl::Buffer kernelBuffer(
        openclManager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * gaussianKernel.size(), gaussianKernel.data());

    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, kernelBuffer);
    kernel.setArg(3, kernelSize);

    // Execute kernel
    cl::CommandQueue& queue =
        openclManager.getQueue(0 /* Assume GPU is pre-selected */);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width, height));
    queue.finish();

    // Retrieve processed data
    cv::Mat processedMat =
        openclManager.readImage2DToMat(outputBuffer, width, height, 0);
    img.setImage(processedMat);
}

void GaussianBlurStepCL::precomputeKernel()
{
    int halfSize = kernelSize / 2;
    gaussianKernel.resize(kernelSize * kernelSize);
    float sum = 0.0f;

    for (int y = -halfSize; y <= halfSize; ++y)
    {
        for (int x = -halfSize; x <= halfSize; ++x)
        {
            float value = exp(-(x * x + y * y) / (2 * sigma * sigma)) /
                          (2 * M_PI * sigma * sigma);
            gaussianKernel[(y + halfSize) * kernelSize + (x + halfSize)] =
                value;
            sum += value;
        }
    }

    for (auto& val : gaussianKernel)
    {
        val /= sum;
    }
}
