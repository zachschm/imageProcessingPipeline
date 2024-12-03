#ifndef ROTATIONFLIPSTEP_H
#define ROTATIONFLIPSTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/imgproc.hpp>

class RotationFlipStep : public ProcessingStep
{
 public:
    RotationFlipStep(double angle, int flipCode)
        : angle(angle)
        , flipCode(flipCode)
    {
    }

    void process(Image& img) override
    {
        cv::Mat rotatedImage;

        // Rotate
        if (angle != 0.0)
        {
            cv::Point2f center(img.getImage().cols / 2.0,
                               img.getImage().rows / 2.0);
            cv::Mat rotationMatrix =
                cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(img.getImage(), rotatedImage, rotationMatrix,
                           img.getImage().size());
        }
        else
        {
            rotatedImage = img.getImage();
        }

        // Flip
        cv::Mat finalImage;
        cv::flip(rotatedImage, finalImage, flipCode);

        img.setImage(finalImage);
    }

 private:
    double angle;
    int flipCode;
};

#endif  // ROTATIONFLIPSTEP_H
