#include "SaturationStepCL.h"

SaturationStepCL::SaturationStepCL(OpenCLManager& manager, float scale)
    : manager(manager)
    , scale(scale)
{
    manager.loadKernel("adjustSaturation", "src/opencl/kernels/saturation.cl");
    kernel = manager.getKernel("adjustSaturation");
}

void SaturationStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();

    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert image to HSV
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    int width = hsvImage.cols;
    int height = hsvImage.rows;

    // Create OpenCL buffers
    cl::Image2D inputBuffer =
        manager.createImage2DFromMat(hsvImage, 0 /* Assume GPU pre-selected */);
    cl::Image2D outputBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                             cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), width,
                             height);

    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, scale);

    // Execute kernel
    cl::CommandQueue& queue = manager.getQueue(0 /* Assume GPU pre-selected */);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width, height));
    queue.finish();

    // Retrieve processed data
    cv::Mat processedMat =
        manager.readImage2DToMat(outputBuffer, width, height, 0);

    // Convert back to BGR
    cv::Mat outputImage;
    cv::cvtColor(processedMat, outputImage, cv::COLOR_HSV2BGR);

    img.setImage(outputImage);
}
