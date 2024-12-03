#ifndef PIPELINE_H
#define PIPELINE_H

#include "ProcessingStep.h"
#include <memory>
#include <vector>

class Pipeline
{
 public:
    // Change the parameter to accept a unique_ptr
    void addStep(std::unique_ptr<ProcessingStep> step)
    {
        steps.push_back(std::move(step));
    }

    void run(Image& img)
    {
        for (const auto& step : steps)
        {
            step->process(img);
        }
    }

 private:
    std::vector<std::unique_ptr<ProcessingStep>> steps;  // Store unique_ptrs
};

#endif  // PIPELINE_H
