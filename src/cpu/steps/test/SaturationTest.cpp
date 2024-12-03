#include "Image.h"
#include "SaturationStep.h"
#include <gtest/gtest.h>

// Test case for Saturation Step
TEST(SaturationTest, SaturationEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 color gradient)
    cv::Mat originalImage(100, 100, CV_8UC3);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            originalImage.at<cv::Vec3b>(i, j) =
                cv::Vec3b(j, 0, 255 - j);  // Blue to Red gradient
        }
    }
    Image img(originalImage);

    // Convert original image to HSV
    cv::Mat hsvOriginal;
    cv::cvtColor(originalImage, hsvOriginal, cv::COLOR_BGR2HSV);

    // Create the SaturationStep (increase saturation)
    SaturationStep saturationStep(1.5);  // Increase saturation by 50%

    // Apply saturation adjustment
    saturationStep.process(img);

    // Get the saturated image
    cv::Mat saturatedImage = img.getImage();

    // Convert the saturated image to HSV for comparison
    cv::Mat hsvSaturated;
    cv::cvtColor(saturatedImage, hsvSaturated, cv::COLOR_BGR2HSV);

    // Verify the saturation channel (channel 1) has been scaled by 1.5
    uchar originalSaturation = hsvOriginal.at<cv::Vec3b>(50, 50)[1];
    uchar expectedSaturation =
        cv::saturate_cast<uchar>(1.5f * originalSaturation);
    uchar actualSaturation = hsvSaturated.at<cv::Vec3b>(50, 50)[1];

    // Output for debugging
    std::cout << "Original Saturation: " << (int)originalSaturation
              << std::endl;
    std::cout << "Expected Saturation: " << (int)expectedSaturation
              << std::endl;
    std::cout << "Actual Saturation: " << (int)actualSaturation << std::endl;

    // Verify saturation value
    ASSERT_EQ(actualSaturation, expectedSaturation);
}

// Test case for zero saturation
TEST(SaturationTest, ZeroSaturationEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 color gradient)
    cv::Mat originalImage(100, 100, CV_8UC3);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            originalImage.at<cv::Vec3b>(i, j) =
                cv::Vec3b(j, 0, 255 - j);  // Blue to Red gradient
        }
    }
    Image img(originalImage);

    // Create the SaturationStep (zero saturation)
    SaturationStep saturationStep(0.0f);  // Set saturation to 0%

    // Apply saturation adjustment
    saturationStep.process(img);

    // Get the desaturated image
    cv::Mat desaturatedImage = img.getImage();

    // Check if the output image is grayscale by comparing R, G, B channels
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            ASSERT_EQ(desaturatedImage.at<cv::Vec3b>(i, j)[0],
                      desaturatedImage.at<cv::Vec3b>(i, j)[1]);
            ASSERT_EQ(desaturatedImage.at<cv::Vec3b>(i, j)[1],
                      desaturatedImage.at<cv::Vec3b>(i, j)[2]);
        }
    }
}
