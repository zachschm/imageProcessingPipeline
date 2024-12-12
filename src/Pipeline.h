#ifndef PIPELINE_H
#define PIPELINE_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <memory>
#include <vector>

class Pipeline
{
 public:
    explicit Pipeline(OpenCLManager* oclManager = nullptr);

    // Add a processing step to the pipeline
    void addStep(std::unique_ptr<ProcessingStep> step);

    // Run the pipeline on a single image
    void run(Image& img);

 private:
    OpenCLManager* openclManager;  // Manager for OpenCL resources
    std::vector<std::unique_ptr<ProcessingStep>>
        steps;  // List of processing steps
};

#endif  // PIPELINE_H
