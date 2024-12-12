#include "SobelEdgeStepCL.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>

SobelEdgeStepCL::SobelEdgeStepCL(OpenCLManager& manager)
    : manager(manager)
{
    std::string kernelPath = std::string(std::getenv("PROJECT_ROOT")) + "/src/openCL/kernels/sobel_edge.cl";
    manager.loadKernel("sobelEdge", kernelPath);
    kernel = manager.getKernel("sobelEdge");
}

void SobelEdgeStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    int width = grayscaleImage.cols;
    int height = grayscaleImage.rows;

    // Create OpenCL buffers
    cl::Image2D inputBuffer = manager.createImage2DFromMat(
        grayscaleImage, 0 /* Assume GPU pre-selected */);
    cl::Image2D outputBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                             cl::ImageFormat(CL_R, CL_FLOAT), width, height);

    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);

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
