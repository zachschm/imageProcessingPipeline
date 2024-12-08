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
    ImageSplitter splitter;
    ImageMerger merger;
    std::vector<Image> allParts;
    for (auto& img : images)
    {
        auto parts = splitter.split(img.getImage(), 4);
        allParts.insert(allParts.end(), parts.begin(), parts.end());
    }

    std::vector<Image> processedParts(4);
    std::vector<std::thread> gpuThreads;

    for (int gpuIndex = 0; gpuIndex < 4; ++gpuIndex)
    {
        gpuThreads.emplace_back(
            [&, gpuIndex]()
            {
                std::vector<Image> gpuParts;
                for (size_t i = gpuIndex; i < allParts.size(); i += 4)
                {
                    gpuParts.push_back(allParts[i]);
                }
                pipeline.runBatch(gpuParts);
            });
    }

    for (auto& thread : gpuThreads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        auto parts = std::vector<Image>(allParts.begin() + i * 4,
                                        allParts.begin() + (i + 1) * 4);
        images[i] = merger.merge(parts);
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
