#ifndef PROCESSINGSTEPFACTORY_H
#define PROCESSINGSTEPFACTORY_H

#include "BrightnessStep.h"
#include "CenterCropStep.h"
#include "ContrastStep.h"
#include "GaussianBlurStep.h"
#include "GaussianBlurStepCL.h"
#include "GrayscaleStep.h"
#include "GrayscaleStepCL.h"
#include "HistogramEqualizationStep.h"
#include "HistogramEqualizationStepCL.h" 
#include "ProcessingParameters.h"
#include "ResizeStep.h"
#include "ResizeStepCL.h"
#include "RotationFlipStep.h"
#include "SaturationStep.h"
#include "SaturationStepCL.h"
#include "SharpeningStep.h"
#include "SharpeningStepCL.h"
#include "SobelEdgeStep.h"
#include "SobelEdgeStepCL.h"
#include "ThresholdStep.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

class ProcessingStepFactory
{
 public:
    using StepCreator = std::function<std::unique_ptr<ProcessingStep>(
        const ProcessingParameters&)>;

    ProcessingStepFactory(OpenCLManager& manager)
        : openclManager(manager)
    {
        const char* mode = std::getenv("PROCESSING_MODE");
        std::string processingMode = mode ? std::string(mode) : "gpu";
        if (manager.getDeviceCount() < 4)
        {
            processingMode = "cpu";
        }

        if (processingMode == "gpu")
        {
            stepCreators["grayscale"] = [this](const ProcessingParameters&)
            { return std::make_unique<GrayscaleStepCL>(openclManager); };

            stepCreators["resize"] = [this](const ProcessingParameters& params)
            {
                return std::make_unique<ResizeStepCL>(
                    openclManager, params.resizeParams.width,
                    params.resizeParams.height);
            };

            stepCreators["gaussian_blur"] =
                [this](const ProcessingParameters& params)
            {
                return std::make_unique<GaussianBlurStepCL>(
                    openclManager, params.gaussianBlurParams.kernelSize,
                    params.gaussianBlurParams.sigmaX);
            };

            stepCreators["edge_detection"] = [this](const ProcessingParameters&)
            { return std::make_unique<SobelEdgeStepCL>(openclManager); };

            stepCreators["histogram_equalization"] =
                [this](const ProcessingParameters&) {
                    return std::make_unique<HistogramEqualizationStepCL>(
                        openclManager);
                };

            stepCreators["sharpening"] = [this](const ProcessingParameters&)
            { return std::make_unique<SharpeningStepCL>(openclManager, 3); };
        }
        else
        {
            // CPU steps
            stepCreators["grayscale"] = [](const ProcessingParameters&)
            { return std::make_unique<GrayscaleStep>(); };

            stepCreators["resize"] = [](const ProcessingParameters& params)
            {
                return std::make_unique<ResizeStep>(params.resizeParams.width,
                                                    params.resizeParams.height);
            };
            stepCreators["gaussian_blur"] =
                [](const ProcessingParameters& params)
            {
                return std::make_unique<GaussianBlurStep>(
                    params.gaussianBlurParams.kernelSize,
                    params.gaussianBlurParams.sigmaX,
                    params.gaussianBlurParams.sigmaY);
            };

            stepCreators["edge_detection"] = [](const ProcessingParameters&)
            { return std::make_unique<SobelEdgeStep>(); };

            stepCreators["saturation_adjustment"] =
                [](const ProcessingParameters& params) {
                    return std::make_unique<SaturationStep>(
                        params.saturationParams.scale);
                };

            stepCreators["sharpening"] = [](const ProcessingParameters&)
            { return std::make_unique<SharpeningStep>(); };

            stepCreators["histogram_equalization"] =
                [](const ProcessingParameters&)
            { return std::make_unique<HistogramEqualizationStep>(); };
        }

        stepCreators["center_crop"] = [](const ProcessingParameters& params)
        {
            return std::make_unique<CenterCropStep>(params.cropParams.width,
                                                    params.cropParams.height);
        };

        stepCreators["brightness"] = [](const ProcessingParameters& params)
        {
            return std::make_unique<BrightnessStep>(
                params.brightnessParams.brightness);
        };

        stepCreators["contrast"] = [](const ProcessingParameters& params)
        { return std::make_unique<ContrastStep>(params.contrastParams.alpha); };

        stepCreators["rotation_flip"] = [](const ProcessingParameters& params)
        {
            return std::make_unique<RotationFlipStep>(
                params.rotationFlipParams.angle,
                params.rotationFlipParams.flipCode);
        };

        stepCreators["threshold"] = [](const ProcessingParameters& params) {
            return std::make_unique<ThresholdStep>(
                params.thresholdParams.threshold);
        };
    }

    std::unique_ptr<ProcessingStep>
    createProcessingStep(const std::string& processType,
                         const ProcessingParameters& params)
    {
        auto it = stepCreators.find(processType);
        if (it != stepCreators.end())
        {
            return it->second(params);
        }
        return nullptr;
    }

 private:
    OpenCLManager& openclManager;
    std::unordered_map<std::string, StepCreator> stepCreators;
};

#endif  // PROCESSINGSTEPFACTORY_H
