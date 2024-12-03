#include "Pipeline.h"
#include "ImageMerger.h"
#include "ImageSplitter.h"
#include <barrier>
#include <future>
#include <iostream>
#include <stdexcept>

Pipeline::Pipeline(OpenCLManager* oclManager)
    : openclManager(oclManager)
{
}

void Pipeline::addStep(std::unique_ptr<ProcessingStep> step)
{
    steps.push_back(std::move(step));
}

void Pipeline::run(Image& img)
{
    for (const auto& step : steps)
    {
        step->process(img);
    }
}

void Pipeline::runBatch(std::vector<Image>& images)
{
    if (!openclManager)
    {
        throw std::runtime_error(
            "OpenCLManager is not initialized for GPU processing.");
    }

    int gpuCount = openclManager->getDeviceCount();
    if (gpuCount != 4)
    {
        throw std::runtime_error(
            "Expected exactly 4 GPUs for multi-GPU execution.");
    }

    size_t imageCount = images.size();
    size_t chunkSize = imageCount / gpuCount;
    std::vector<std::future<void>> futures;

    // Synchronization barrier for all GPUs
    std::barrier syncBarrier(gpuCount, []()
                             { std::cout << "All GPU tasks synchronized.\n"; });

    // Launch processing on each GPU
    for (int gpuIndex = 0; gpuIndex < gpuCount; ++gpuIndex)
    {
        size_t startIdx = gpuIndex * chunkSize;
        size_t endIdx =
            (gpuIndex == gpuCount - 1) ? imageCount : startIdx + chunkSize;

        futures.push_back(std::async(
            std::launch::async,
            [this, &images, startIdx, endIdx, gpuIndex, &syncBarrier]()
            {
                try
                {
                    for (size_t i = startIdx; i < endIdx; ++i)
                    {
                        // Split the image into parts for processing
                        std::vector<Image> parts =
                            ImageSplitter::split(images[i], 4);

                        // Process each part using pipeline steps
                        for (Image& part : parts)
                        {
                            for (const auto& step : steps)
                            {
                                step->process(part, gpuIndex);
                            }
                        }

                        // Merge the processed parts back into the final image
                        images[i] = ImageMerger::merge(parts);

                        std::cout << "GPU " << gpuIndex << " processed image "
                                  << i << ".\n";
                    }

                    // Ensure GPU tasks are complete
                    openclManager->getQueue(gpuIndex).finish();
                    syncBarrier.arrive_and_wait();
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Error on GPU " << gpuIndex << ": " << e.what()
                              << "\n";
                    throw;
                }
            }));
    }

    // Wait for all GPU threads to finish
    for (auto& fut : futures)
    {
        fut.get();
    }

    std::cout << "All images processed successfully.\n";
}
