#include "GaussianBlurStep.h"
#include "Image.h"
#include <gtest/gtest.h>

// Test case for Gaussian Blur Step
TEST(GaussianBlurTest, BlurEffectIsAppliedCorrectly)
{
    int kernelSize = 5;   // Example kernel size
    double sigmaX = 1.0;  // Standard deviation in the X direction
    double sigmaY = 1.0;  // Standard deviation in the Y direction

    // Create a gradient test image
    cv::Mat testImage = cv::Mat::zeros(100, 100, CV_8UC3);
    for (int i = 0; i < testImage.rows; ++i)
    {
        for (int j = 0; j < testImage.cols; ++j)
        {
            testImage.at<cv::Vec3b>(i, j) =
                cv::Vec3b(i, j, (i + j) / 2);  // Simple gradient pattern
        }
    }

    Image img(testImage);

    // Create the GaussianBlurStep
    GaussianBlurStep blurStep(kernelSize, sigmaX, sigmaY);

    // Apply the blur
    blurStep.process(img);

    // Get the processed image
    cv::Mat blurredImage = img.getImage();

    // Test: Ensure the image has been modified (blurred)
    ASSERT_FALSE(blurredImage.empty()) << "Blurred image should not be empty.";

    // Test: The resulting blurred image should still have the same dimensions
    ASSERT_EQ(blurredImage.rows, testImage.rows);
    ASSERT_EQ(blurredImage.cols, testImage.cols);

    // Test: Verify that the blur was applied by checking pixel differences
    cv::Scalar originalPixelValue = cv::mean(testImage);
    cv::Scalar blurredPixelValue = cv::mean(blurredImage);

    // The blurred image should have different mean pixel values than the
    // original
    ASSERT_NE(originalPixelValue, blurredPixelValue)
        << "Blurred image should differ from the original image.";

    // Optionally, you could also check specific pixel values if necessary.
}
