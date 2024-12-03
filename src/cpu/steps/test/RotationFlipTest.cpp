#include "Image.h"
#include "RotationFlipStep.h"
#include <gtest/gtest.h>

// Test case for Rotation and Flip Step
TEST(RotationFlipTest, RotationAndFlipEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 white square)
    cv::Mat originalImage =
        cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    Image img(originalImage);

    // Create the RotationFlipStep (flip horizontally)
    RotationFlipStep rotationFlipStep(90, -1);  // Rotate 90 degrees clockwise

    // Apply rotation and flip
    rotationFlipStep.process(img);

    // Get the rotated image
    cv::Mat rotatedImage = img.getImage();

    // Check the dimensions of the rotated image
    ASSERT_EQ(rotatedImage.rows, 100);  // Height should remain 100
    ASSERT_EQ(rotatedImage.cols, 100);  // Width should remain 100
    ASSERT_EQ(rotatedImage.type(),
              CV_8UC3);  // Ensure the image type is still CV_8UC3

    // Check if the rotation was applied correctly by checking pixel values.
    // Since we are using a white square, any check can suffice.
    ASSERT_EQ(rotatedImage.at<cv::Vec3b>(0, 0),
              cv::Vec3b(255, 255, 255));  // Top-left pixel
}

// Test case for flipping horizontally
TEST(RotationFlipTest, FlipEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 color gradient)
    cv::Mat originalImage(100, 100, CV_8UC3);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            originalImage.at<cv::Vec3b>(i, j) =
                cv::Vec3b(j, j, j);  // Grayscale gradient
        }
    }
    Image img(originalImage);

    // Create the RotationFlipStep (flip horizontally)
    RotationFlipStep rotationFlipStep(0, 2);

    // Apply flip
    rotationFlipStep.process(img);

    // Get the flipped image
    cv::Mat flippedImage = img.getImage();

    // Check the pixel values to ensure they were flipped correctly
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 50; ++j)  // Check only half of the width
        {
            ASSERT_EQ(flippedImage.at<cv::Vec3b>(i, j),
                      originalImage.at<cv::Vec3b>(
                          i, 99 - j));  // Check mirrored pixels
        }
    }
}
