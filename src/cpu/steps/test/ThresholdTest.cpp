#include "Image.h"
#include "ThresholdStep.h"
#include <gtest/gtest.h>

// Test case for Threshold
// Test case for Threshold Step with Otsu's Method
TEST(ThresholdTest, ThresholdEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 with a gradient)
    cv::Mat originalImage(100, 100, CV_8UC1);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            originalImage.at<uchar>(i, j) =
                static_cast<uchar>((i + j) / 2);  // Create a gradient
        }
    }
    Image img(originalImage);

    // Create the ThresholdStep with Otsu's threshold value
    ThresholdStep thresholdStep(128);  // Example value, won't be used with Otsu

    // Apply threshold
    thresholdStep.process(img);

    // Get the processed image
    cv::Mat thresholdedImage = img.getImage();

    // Calculate the expected threshold value using Otsu's method
    double otsuThreshold = cv::threshold(originalImage, originalImage, 0, 255,
                                         cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Check that pixels below the Otsu's threshold are set to 0 and above are
    // set to 255
    for (int i = 0; i < thresholdedImage.rows; ++i)
    {
        for (int j = 0; j < thresholdedImage.cols; ++j)
        {
            if (originalImage.at<uchar>(i, j) < otsuThreshold)
            {
                ASSERT_EQ(thresholdedImage.at<uchar>(i, j), 0);
            }
            else
            {
                ASSERT_EQ(thresholdedImage.at<uchar>(i, j), 255);
            }
        }
    }
}

// Test case for threshold with a uniform image
TEST(ThresholdTest, NoChangeOnUniformImage)
{
    // Create a simple uniform image (100x100 with a single color)
    cv::Mat originalImage(100, 100, CV_8UC1, cv::Scalar(200));  // Gray image
    Image img(originalImage);

    // Create the ThresholdStep with a threshold value
    int thresholdValue = 128;  // Example threshold value
    ThresholdStep thresholdStep(thresholdValue);

    // Apply threshold
    thresholdStep.process(img);

    // Get the processed image
    cv::Mat thresholdedImage = img.getImage();

    // Check that all pixels are set to 255 (since they are above the threshold)
    for (int i = 0; i < thresholdedImage.rows; ++i)
    {
        for (int j = 0; j < thresholdedImage.cols; ++j)
        {
            ASSERT_EQ(thresholdedImage.at<uchar>(i, j), 255);
        }
    }
}
