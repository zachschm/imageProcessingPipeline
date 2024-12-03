#ifndef PROCESSINGSTEPFACTORY_H
#define PROCESSINGSTEPFACTORY_H

#include "BrightnessStep.h"
#include "CenterCropStep.h"
#include "ContrastStep.h"
#include "GaussianBlurStep.h"
#include "GrayscaleStep.h"
#include "HistogramEqualizationStep.h"
#include "OpenCLGrayscaleStep.h"
#include "ProcessingParameters.h"
#include "ResizeStep.h"
#include "RotationFlipStep.h"
#include "SaturationStep.h"
#include "SharpeningStep.h"
#include "SobelEdgeStep.h"
#include "ThresholdStep.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

class ProcessingStepFactory
{
 public:
    // Using a function pointer type for creating processing steps
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
            stepCreators["grayscale"] = [this]()
            { return std::make_unique<OpenCLGrayscaleStep>(openclManager); };

            stepCreators["resize"] = [this]()
            {
                return std::make_unique<OpenCLResizeStep>(openclManager,
                                                          gpuIndex, 800, 600);
            };

            stepCreators["gaussian_blur"] = [this]()
            {
                return std::make_unique<OpenCLGaussianBlurStep>(
                    openclManager, 5, 1.0f);  // Example kernel size and sigma
            };

            stepCreators["edge_detection"] = [this]()
            { return std::make_unique<OpenCLSobelEdgeStep>(openclManager); };

            stepCreators["histogram_equalization"] = [this]() {
                return std::make_unique<OpenCLHistogramEqualizationStep>(
                    openclManager);
            };

            stepCreators["sharpening"] = [this]()
            { return std::make_unique<OpenCLSharpeningStep>(openclManager); };
        }
        else
        {

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

            stepCreators["sharpening"] = [](const ProcessingParameters& params)
            { return std::make_unique<SharpeningStep>(); };

            stepCreators["histogram_equalization"] =
                [](const ProcessingParameters& params)
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

    // Function to create a processing step based on the processType
    std::unique_ptr<ProcessingStep>
    createProcessingStep(const std::string& processType,
                         const ProcessingParameters& params)
    {
        auto it = stepCreators.find(processType);
        if (it != stepCreators.end())
        {
            return it->second(
                params);  // Call the corresponding creation function
        }
        return nullptr;  // Return nullptr if the processType is not found
    }

 private:
    // Map to store processing type to creation function mapping
    std::unordered_map<std::string, StepCreator> stepCreators;
};

#endif  // PROCESSINGSTEPFACTORY_H
