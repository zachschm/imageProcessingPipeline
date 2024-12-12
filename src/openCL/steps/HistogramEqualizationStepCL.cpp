#include "HistogramEqualizationStepCl.h"

HistogramEqualizationStepCL::HistogramEqualizationStepCL(OpenCLManager& manager)
    : openclManager(manager)
{
    // Load kernel
    openclManager.loadKernel("histogram_equalization",
                             "src/opencl/kernels/histogram_equalization.cl");
    kernel = openclManager.getKernel("histogram_equalization");
}

void HistogramEqualizationStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty() || inputImage.channels() != 1)
    {
        throw std::runtime_error(
            "Input image must be a non-empty grayscale image.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int totalPixels = width * height;

    // Split image into 4 parts for 4 GPUs
    std::vector<cv::Mat> segments = splitImage(inputImage, 4);

    std::vector<cl::Buffer> inputBuffers(4);
    std::vector<cl::Buffer> outputBuffers(4);
    std::vector<cl::Buffer> histBuffers(4);
    std::vector<cl::Buffer> cdfBuffers(4);
    std::vector<std::vector<uchar>> outputData(4);

    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        int segmentHeight = segments[gpuIndex].rows;

        inputBuffers[gpuIndex] = cl::Buffer(
            openclManager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            static_cast<size_t>(width * segmentHeight),
            segments[gpuIndex].data);

        outputBuffers[gpuIndex] =
            cl::Buffer(openclManager.getContext(), CL_MEM_WRITE_ONLY,
                       width * segmentHeight);

        histBuffers[gpuIndex] =
            cl::Buffer(openclManager.getContext(), CL_MEM_READ_WRITE,
                       sizeof(int) * 256, nullptr);

        cdfBuffers[gpuIndex] =
            cl::Buffer(openclManager.getContext(), CL_MEM_READ_WRITE,
                       sizeof(int) * 256, nullptr);

        kernel.setArg(0, inputBuffers[gpuIndex]);
        kernel.setArg(1, outputBuffers[gpuIndex]);
        kernel.setArg(2, histBuffers[gpuIndex]);
        kernel.setArg(3, cdfBuffers[gpuIndex]);
        kernel.setArg(4, width);
        kernel.setArg(5, segmentHeight);
        kernel.setArg(6, gpuIndex * segmentHeight);
        kernel.setArg(7, width * segmentHeight);

        cl::CommandQueue& queue = openclManager.getQueue(gpuIndex);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(width, segmentHeight));
    }

    // Retrieve results
    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        cl::CommandQueue& queue = openclManager.getQueue(gpuIndex);
        queue.finish();

        int segmentHeight = segments[gpuIndex].rows;
        outputData[gpuIndex].resize(width * segmentHeight);

        queue.enqueueReadBuffer(outputBuffers[gpuIndex], CL_TRUE, 0,
                                width * segmentHeight,
                                outputData[gpuIndex].data());
    }

    // Merge results
    cv::Mat outputImage = mergeImage(outputData, width, height);
    img.setImage(outputImage);
}

std::vector<cv::Mat> HistogramEqualizationStepCL::splitImage(const cv::Mat& img,
                                                             int parts)
{
    int rowsPerPart = img.rows / parts;
    std::vector<cv::Mat> segments;

    for (int i = 0; i < parts; ++i)
    {
        int startRow = i * rowsPerPart;
        int endRow = (i == parts - 1) ? img.rows : startRow + rowsPerPart;
        segments.push_back(img.rowRange(startRow, endRow));
    }

    return segments;
}

cv::Mat HistogramEqualizationStepCL::mergeImage(
    const std::vector<std::vector<uchar>>& segments, int width, int height)
{
    cv::Mat output(height, width, CV_8UC1);
    int currentRow = 0;

    for (const auto& segment : segments)
    {
        cv::Mat segmentMat(segment.size() / width, width, CV_8UC1,
                           const_cast<uchar*>(segment.data()));
        segmentMat.copyTo(
            output.rowRange(currentRow, currentRow + segmentMat.rows));
        currentRow += segmentMat.rows;
    }

    return output;
}
