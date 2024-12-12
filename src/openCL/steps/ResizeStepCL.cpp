#include "ResizeStepCL.h"

ResizeStepCL::ResizeStepCL(OpenCLManager& manager, int newWidth, int newHeight)
    : manager(manager)
    , newWidth(newWidth)
    , newHeight(newHeight)
{
    manager.loadKernel("resize", "src/opencl/kernels/resize.cl");
    kernel = manager.getKernel("resize");
}

void ResizeStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Create OpenCL buffers
    cl::Image2D inputImageBuffer =
        manager.createImage2DFromMat(inputImage, 0 /* GPU pre-selected */);
    cl::Image2D outputImageBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                                  cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                                  newWidth, newHeight);

    float scaleX = static_cast<float>(newWidth) / width;
    float scaleY = static_cast<float>(newHeight) / height;

    // Set kernel arguments
    kernel.setArg(0, inputImageBuffer);
    kernel.setArg(1, outputImageBuffer);
    kernel.setArg(2, scaleX);
    kernel.setArg(3, scaleY);

    // Execute kernel
    cl::CommandQueue& queue = manager.getQueue(0 /* GPU pre-selected */);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(newWidth, newHeight));
    queue.finish();

    // Retrieve the resized image
    cv::Mat resizedMat =
        manager.readImage2DToMat(outputImageBuffer, newWidth, newHeight, 0);
    img.setImage(resizedMat);
}
