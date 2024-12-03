#include "SharpeningStepCL.h"
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>

SharpeningStepCL::SharpeningStepCL(OpenCLManager& manager, int kernelSize)
    : manager(manager)
    , kernelSize(kernelSize)
{
    manager.loadKernel("sharpening", "src/opencl/kernels/sharpening.cl");
    kernel = manager.getKernel("sharpening");

    // Define a sharpening kernel
    sharpeningKernel = {0, -1, 0, -1, 5, -1, 0, -1, 0};
}

void SharpeningStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // Split the image for multi-GPU processing
    auto subImages =
        ImageSplitter::split(grayscaleImage, manager.getDeviceCount());
    std::vector<cv::Mat> processedSubImages(subImages.size());

#pragma omp parallel for
    for (int i = 0; i < subImages.size(); ++i)
    {
        cl::Image2D inputBuffer = manager.createImage2DFromMat(subImages[i], i);
        cl::Image2D outputBuffer(manager.getContext(i), CL_MEM_WRITE_ONLY,
                                 cl::ImageFormat(CL_R, CL_FLOAT),
                                 subImages[i].cols, subImages[i].rows);

        cl::Buffer kernelBuffer(
            manager.getContext(i), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * sharpeningKernel.size(), sharpeningKernel.data());

        // Set up kernel
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, kernelBuffer);
        kernel.setArg(3, kernelSize);

        // Execute kernel
        cl::CommandQueue& queue = manager.getQueue(i);
        cl::NDRange globalSize(subImages[i].cols, subImages[i].rows);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
        queue.finish();

        // Retrieve processed sub-image
        processedSubImages[i] = manager.readImage2DToMat(
            outputBuffer, subImages[i].cols, subImages[i].rows, i);
    }

    // Merge processed sub-images
    cv::Mat mergedImage = ImageMerger::merge(
        processedSubImages, grayscaleImage.cols, grayscaleImage.rows);

    // Set the processed image
    img.setImage(mergedImage);
}
