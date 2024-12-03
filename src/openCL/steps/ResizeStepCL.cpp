#include "ResizeStepCL.h"
#include <omp.h>

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
    int width = img.getImage().cols;
    int height = img.getImage().rows;

    // Split the image into 4 parts
    auto subImages = ImageSplitter::split(img, 4);

    std::vector<cv::Mat> resizedParts(4);

    // Process each part in parallel on separate GPUs
#pragma omp parallel for
    for (int i = 0; i < 4; ++i)
    {
        int gpuIndex = i;
        cl::Image2D inputImageBuffer =
            manager.createImage2DFromMat(subImages[i], gpuIndex);
        cl::Image2D outputImageBuffer(
            manager.getContext(gpuIndex), CL_MEM_WRITE_ONLY,
            cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), newWidth / 2, newHeight);

        float scaleX = static_cast<float>(newWidth) / width;
        float scaleY = static_cast<float>(newHeight) / height;

        // Set kernel arguments
        kernel.setArg(0, inputImageBuffer);
        kernel.setArg(1, outputImageBuffer);
        kernel.setArg(2, scaleX);
        kernel.setArg(3, scaleY);

        cl::CommandQueue& queue = manager.getQueue(gpuIndex);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(newWidth / 2, newHeight / 2));
        queue.finish();

        resizedParts[i] = manager.readImage2DToMat(
            outputImageBuffer, newWidth / 2, newHeight / 2, gpuIndex);
    }

    // Merge the resized parts
    cv::Mat resizedImage =
        ImageMerger::merge(resizedParts, newWidth, newHeight);

    // Update the image with the resized data
    img.setImage(resizedImage);
}
