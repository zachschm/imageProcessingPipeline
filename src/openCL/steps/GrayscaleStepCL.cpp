#include "GrayscaleStepCL.h"

GrayscaleStepCL::GrayscaleStepCL(OpenCLManager& manager)
    : openclManager(manager)
{
    // Load and compile the grayscale kernel
    openclManager.loadKernel("grayscale", "src/opencl/kernels/grayscale.cl");
    kernel = openclManager.getKernel("grayscale");
}

void GrayscaleStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty() || inputImage.channels() != 3)
    {
        throw std::runtime_error("Input image must be a non-empty RGB image.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int chunkHeight = height / 4;

    std::vector<cl::Event> events;
    std::vector<cv::Mat> outputChunks(4);
    std::vector<cl::Buffer> outputBuffers(4);

    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        int startRow = gpuIndex * chunkHeight;
        int currentChunkHeight =
            (gpuIndex == 3) ? (height - startRow) : chunkHeight;

        cv::Mat chunk =
            inputImage.rowRange(startRow, startRow + currentChunkHeight);
        size_t chunkSize = chunk.total() * chunk.elemSize();

        cl::Buffer inputBuffer(openclManager.getContext(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               chunkSize, chunk.data);
        cl::Buffer outputBuffer(openclManager.getContext(), CL_MEM_WRITE_ONLY,
                                currentChunkHeight * width);

        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, width);
        kernel.setArg(3, height);
        kernel.setArg(4, startRow);

        auto& queue = openclManager.getQueue(gpuIndex);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(width, currentChunkHeight),
                                   cl::NullRange, nullptr, &event);
        events.push_back(event);

        outputBuffers[gpuIndex] = outputBuffer;
        outputChunks[gpuIndex] = cv::Mat(currentChunkHeight, width, CV_8UC1);
    }

    for (auto& event : events)
    {
        event.wait();
    }

    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        auto& queue = openclManager.getQueue(gpuIndex);
        queue.enqueueReadBuffer(outputBuffers[gpuIndex], CL_TRUE, 0,
                                outputChunks[gpuIndex].total(),
                                outputChunks[gpuIndex].data);
    }

    cv::Mat outputImage(height, width, CV_8UC1);
    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        int startRow = gpuIndex * chunkHeight;
        outputChunks[gpuIndex].copyTo(outputImage.rowRange(
            startRow, startRow + outputChunks[gpuIndex].rows));
    }

    img.setImage(outputImage);
}
