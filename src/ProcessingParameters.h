#ifndef PROCESSINGPARAMETERS_H
#define PROCESSINGPARAMETERS_H

struct ProcessingParameters
{
    struct ResizeParams
    {
        int width = 1920;
        int height = 1080;
    };

    struct CropParams
    {
        int width = 100;
        int height = 100;
    };

    struct BrightnessParams
    {
        int brightness = 50;
    };

    struct ContrastParams
    {
        float alpha = 1.2;  // Default contrast level (no change)
    };

    struct GaussianBlurParams
    {
        int kernelSize = 5;  // Kernel size, must be odd
        double sigmaX = 0;  // Gaussian kernel standard deviation in X direction
        double sigmaY = 0;  // Gaussian kernel standard deviation in Y direction
    };

    struct RotationFlipParams
    {
        double angle = 0.0;  // Rotation angle in degrees
        // Flip code: 0 for vertical, 1 for horizontal, -1 for both
        int flipCode = 0;
    };

    struct SaturationParams
    {
        double scale = 1.5;  // Scale factor for saturation adjustment (1.0
                             // means no change)
    };

    struct ThresholdParams
    {
        int threshold = 128;
    };

    // No optional types; just use default-initialized values
    ResizeParams resizeParams;          // Default: 1920x1080
    CropParams cropParams;              // Default: 0,0 (100x100)
    BrightnessParams brightnessParams;  // Default: 50 brightness
    ContrastParams contrastParams;
    GaussianBlurParams gaussianBlurParams;
    RotationFlipParams rotationFlipParams;
    SaturationParams saturationParams;
    ThresholdParams thresholdParams;
};

#endif  // PROCESSINGPARAMETERS_H
