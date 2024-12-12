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