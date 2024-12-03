#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include "Image.h"
#include "ImageMerger.h"
#include "ImageSplitter.h"
#include "OpenCLManager.h"
#include "Pipeline.h"
#include "ProcessingStepFactory.h"
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class BatchManager
{
 public:
    explicit BatchManager(const std::string& dir, OpenCLManager& oclManager);
    void processBatch();

 private:
    std::string directory;
    OpenCLManager& openclManager;

    void processBatchOnGPU(std::vector<Image>& images, Pipeline& pipeline,
                           const std::string& processType);
    void processBatchOnCPU(std::vector<Image>& images, Pipeline& pipeline,
                           const std::string& processType);
    void saveImages(const std::vector<Image>& images,
                    const std::string& processType);
};

#endif  // BATCHMANAGER_H
