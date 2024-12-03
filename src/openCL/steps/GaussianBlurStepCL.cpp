#include "GaussianBlurStepCl.h"
#include <cmath>
#include <stdexcept>

GaussianBlurStepCl::GaussianBlurStepCL(OpenCLManager& manager, int kernelSize,
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

    // Split image into 4 parts
    auto subImages =
        ImageSplitter::split(inputImage, openclManager.getDeviceCount());
    std::vector<cv::Mat> processedSubImages(subImages.size());

#pragma omp parallel for
    for (int gpuIndex = 0; gpuIndex < subImages.size(); ++gpuIndex)
    {
        const auto& subImage = subImages[gpuIndex];
        int width = subImage.cols;
        int height = subImage.rows;

        // Create OpenCL buffers
        cl::Image2D inputBuffer =
            openclManager.createImage2DFromMat(subImage, gpuIndex);
        cl::Image2D outputBuffer(
            openclManager.getContext(gpuIndex), CL_MEM_WRITE_ONLY,
            cl::ImageFormat(CL_R, CL_FLOAT), width, height);

        cl::Buffer kernelBuffer(openclManager.getContext(gpuIndex),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * gaussianKernel.size(),
                                gaussianKernel.data());

        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, kernelBuffer);
        kernel.setArg(3, kernelSize);

        // Execute kernel
        cl::CommandQueue& queue = openclManager.getQueue(gpuIndex);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(width, height));
        queue.finish();

        // Retrieve processed sub-image
        processedSubImages[gpuIndex] = openclManager.readImage2DToMat(
            outputBuffer, width, height, gpuIndex);
    }

    // Merge processed images
    cv::Mat outputImage = ImageMerger::merge(processedSubImages,
                                             inputImage.cols, inputImage.rows);
    img.setImage(outputImage);
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
