#include "Image.h"
#include "ResizeStep.h"
#include <gtest/gtest.h>

// Test case for Resize Step
TEST(ResizeTest, ResizeEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 white square)
    cv::Mat originalImage =
        cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    Image img(originalImage);

    // Create the ResizeStep with desired dimensions
    ResizeStep resizeStep(50, 50);  // Resize to 50x50 pixels

    // Apply resizing
    resizeStep.process(img);

    // Get the resized image
    cv::Mat resizedImage = img.getImage();

    // Check the dimensions of the resized image
    ASSERT_EQ(resizedImage.rows, 50);  // Height should be 50
    ASSERT_EQ(resizedImage.cols, 50);  // Width should be 50
    ASSERT_EQ(resizedImage.type(),
              CV_8UC3);  // Ensure the image type is still CV_8UC3
}
