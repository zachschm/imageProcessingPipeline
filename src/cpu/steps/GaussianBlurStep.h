#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>

class GaussianBlurStep : public ProcessingStep
{
 public:
    GaussianBlurStep(int kernelSize, double sigmaX, double sigmaY)
        : kernelSize(kernelSize)
        , sigmaX(sigmaX)
        , sigmaY(sigmaY)
    {
    }

    void process(Image& img) override
    {
        if (!img.getImage().empty())
        {
            cv::Mat blurredImg;
            cv::GaussianBlur(img.getImage(), blurredImg,
                             cv::Size(kernelSize, kernelSize), sigmaX, sigmaY);
            img.setImage(blurredImg);
        }
    }

 private:
    int kernelSize;
    double sigmaX;
    double sigmaY;
};
