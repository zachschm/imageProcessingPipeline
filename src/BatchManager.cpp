#include "BatchManager.h"

BatchManager::BatchManager(const std::string& dir, OpenCLManager& oclManager)
    : directory(dir)
    , openclManager(oclManager)
{
}

void BatchManager::processBatch()
{
    const char* mode = std::getenv("PROCESSING_MODE");
    std::string processingMode = mode ? std::string(mode) : "gpu";

    ProcessingStepFactory stepFactory(openclManager);

    for (const auto& dirEntry : std::filesystem::directory_iterator(directory))
    {
        if (dirEntry.is_directory())
        {
            std::string processType = dirEntry.path().filename().string();
            std::cout << "Processing directory: " << processType << std::endl;

            ProcessingParameters params;
            Pipeline pipeline(&openclManager);

            auto processingStep =
                stepFactory.createProcessingStep(processType, params);
            if (!processingStep)
            {
                std::cerr << "Unknown processing type: " << processType
                          << std::endl;
                continue;
            }

            pipeline.addStep(std::move(processingStep));

            std::vector<Image> images;
            for (const auto& entry :
                 std::filesystem::directory_iterator(dirEntry.path()))
            {
                if (entry.is_regular_file() &&
                    entry.path().filename().string()[0] != '.')
                {
                    Image img;
                    if (img.load(entry.path().string()))
                    {
                        images.push_back(std::move(img));
                    }
                    else
                    {
                        std::cerr << "Error loading image: " << entry.path()
                                  << std::endl;
                    }
                }
            }

            if (processingMode == "gpu")
            {
                processBatchOnGPU(images, pipeline, processType);
            }
            else
            {
                processBatchOnCPU(images, pipeline, processType);
            }
        }
    }
}

void BatchManager::processBatchOnGPU(std::vector<Image>& images,
                                     Pipeline& pipeline,
                                     const std::string& processType)
{
    // Determine the number of available GPUs
    int numGPUs = openclManager.getDeviceCount();
    std::vector<std::thread> gpuThreads;

    // Divide images among GPUs and process
    for (int gpuIndex = 0; gpuIndex < numGPUs; ++gpuIndex)
    {
        gpuThreads.emplace_back(
            [&, gpuIndex]()
            {
                std::vector<Image> gpuSubset;
                for (size_t i = gpuIndex; i < images.size(); i += numGPUs)
                {
                    gpuSubset.push_back(images[i]);
                }

                for (auto& img : gpuSubset)
                {
                    pipeline.run(img);
                }
            });
    }

    for (auto& thread : gpuThreads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    saveImages(images, processType);
}

void BatchManager::processBatchOnCPU(std::vector<Image>& images,
                                     Pipeline& pipeline,
                                     const std::string& processType)
{
    for (auto& img : images)
    {
        pipeline.run(img);
    }
    saveImages(images, processType);
}

void BatchManager::saveImages(const std::vector<Image>& images,
                              const std::string& processType)
{
    std::filesystem::create_directories("output/" + processType);
    for (size_t i = 0; i < images.size(); ++i)
    {
        std::string outputFilePath =
            "output/" + processType + "/image_" + std::to_string(i) + ".png";
        if (!images[i].save(outputFilePath))
        {
            std::cerr << "Error saving image: " << outputFilePath << std::endl;
        }
    }
}
