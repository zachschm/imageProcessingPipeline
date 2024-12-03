#ifndef CONTRASTSTEP_H
#define CONTRASTSTEP_H

#include "Image.h"
#include "ProcessingStep.h"

class ContrastStep : public ProcessingStep
{
 public:
    explicit ContrastStep(float alpha)
        : alpha_(alpha)
    {
    }

    void process(Image& img) override
    {
        cv::Mat newImage;
        img.getImage().convertTo(newImage, -1, alpha_, 0);  // Adjust contrast
        img.setImage(newImage);
    }

 private:
    float alpha_;  // Contrast level
};

#endif  // CONTRASTSTEP_H
