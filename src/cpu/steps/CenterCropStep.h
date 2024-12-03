#ifndef CENTERCROPSTEP_H
#define CENTERCROPSTEP_H

#include "Image.h"
#include "ProcessingStep.h"

class CenterCropStep : public ProcessingStep
{
 public:
    CenterCropStep(int targetWidth, int targetHeight)
        : targetWidth(targetWidth)
        , targetHeight(targetHeight)
    {
    }

    void process(Image& img) override
    {
        cv::Mat original = img.getImage();
        if (original.empty())
        {
            return;
        }

        int width = original.cols;
        int height = original.rows;

        // Calculate center point
        int centerX = width / 2;
        int centerY = height / 2;

        // Calculate crop boundaries
        int startX = std::max(0, centerX - targetWidth / 2);
        int startY = std::max(0, centerY - targetHeight / 2);
        int endX = std::min(width, startX + targetWidth);
        int endY = std::min(height, startY + targetHeight);

        // Perform the crop
        cv::Rect cropRegion(startX, startY, endX - startX, endY - startY);
        cv::Mat croppedImage = original(cropRegion);

        // Set the cropped image in the Image object
        img.setImage(croppedImage);
    }

 private:
    int targetWidth;
    int targetHeight;
};

#endif  // CENTERCROPSTEP_H
