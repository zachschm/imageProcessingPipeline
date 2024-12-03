#include "Image.h"
#include "ProcessingStep.h"

class BrightnessStep : public ProcessingStep
{
 public:
    BrightnessStep(int brightnessVal)
        : brightness(brightnessVal)
    {
    }

    void process(Image& img) override
    {
        if (!img.getImage().empty())
        {
            cv::Mat image = img.getImage();
            cv::Mat result;

            // Add brightness value to each pixel, ensuring we don't overflow or
            // underflow
            image.convertTo(result, -1, 1, brightness);

            img.setImage(result);
        }
    }

 private:
    int brightness;  // Brightness adjustment value
};
