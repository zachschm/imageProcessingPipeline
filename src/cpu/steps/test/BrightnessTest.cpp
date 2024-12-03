#include "BrightnessStep.h"
#include "Image.h"
#include <gtest/gtest.h>

// This main function runs all tests.
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(BrightnessTest, IncreaseBrightness)
{
    // Create a dummy image (3x3 RGB)
    cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC3);
    Image image(img);

    // Apply BrightnessStep with a positive adjustment
    BrightnessStep brightnessStep(50);
    brightnessStep.process(image);

    // Verify that the brightness increased (value should be at least greater
    // than 0)
    cv::Mat processedImage = image.getImage();
    ASSERT_GT(cv::mean(processedImage)[0],
              0);  // Check mean of image channels > 0
}

TEST(BrightnessTest, DecreaseBrightness)
{
    // Create a dummy image (3x3 RGB, all pixels white)
    cv::Mat img = cv::Mat::ones(3, 3, CV_8UC3) * 255;
    Image image(img);

    // Apply BrightnessStep with a negative adjustment
    BrightnessStep brightnessStep(-50);
    brightnessStep.process(image);

    // Verify that the brightness decreased (value should be less than 255)
    cv::Mat processedImage = image.getImage();
    ASSERT_LT(cv::mean(processedImage)[0],
              255);  // Check mean of image channels < 255
}

TEST(BrightnessTest, NoChangeBrightness)
{
    // Create a dummy image (3x3 RGB)
    cv::Mat img = cv::Mat::ones(3, 3, CV_8UC3) * 127;
    Image image(img);

    // Apply BrightnessStep with zero adjustment
    BrightnessStep brightnessStep(0);
    brightnessStep.process(image);

    // Verify that the brightness remained the same
    cv::Mat processedImage = image.getImage();
    ASSERT_EQ(cv::mean(processedImage)[0],
              127);  // Check mean of image channels == 127
}
