#include "SharpeningStepCL.h"
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

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Convert image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // Create OpenCL buffers
    cl::Image2D inputBuffer = manager.createImage2DFromMat(
        grayscaleImage, 0 /* Assume GPU pre-selected */);
    cl::Image2D outputBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                             cl::ImageFormat(CL_R, CL_FLOAT), width, height);

    cl::Buffer kernelBuffer(
        manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * sharpeningKernel.size(), sharpeningKernel.data());

    // Set up kernel
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, kernelBuffer);
    kernel.setArg(3, kernelSize);

    // Execute kernel
    cl::CommandQueue& queue = manager.getQueue(0 /* Assume GPU pre-selected */);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width, height));
    queue.finish();

    // Retrieve processed data
    cv::Mat processedMat =
        manager.readImage2DToMat(outputBuffer, width, height, 0);

    // Update the image with the processed data
    img.setImage(processedMat);
}
