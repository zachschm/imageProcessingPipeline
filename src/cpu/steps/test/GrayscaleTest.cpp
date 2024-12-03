#include "GrayscaleStep.h"
#include "Image.h"
#include <gtest/gtest.h>

// Test case for Grayscale Step
TEST(GrayscaleTest, GrayscaleEffectIsAppliedCorrectly)
{
    // Create a colored image object with some test data (100x100 white image)
    cv::Mat colorImage = cv::Mat(100, 100, CV_8UC3,
                                 cv::Scalar(255, 255, 255));  // White RGB image
    Image img(colorImage);

    // Create the GrayscaleStep
    GrayscaleStep grayscaleStep;

    // Apply the grayscale conversion
    grayscaleStep.process(img);

    // Get the processed grayscale image
    cv::Mat grayscaleImage = img.getImage();

    // Test: Ensure the grayscale image is not empty
    ASSERT_FALSE(grayscaleImage.empty())
        << "Grayscale image should not be empty.";

    // Test: The grayscale image should have the same number of rows and columns
    ASSERT_EQ(grayscaleImage.rows, colorImage.rows);
    ASSERT_EQ(grayscaleImage.cols, colorImage.cols);

    // Test: Ensure the grayscale image has only one channel (since it's
    // grayscale)
    ASSERT_EQ(grayscaleImage.channels(), 1)
        << "Grayscale image should have 1 channel.";

    // Test: Ensure the pixel values in the grayscale image are valid
    cv::Scalar grayscalePixelValue = cv::mean(grayscaleImage);
    ASSERT_GE(grayscalePixelValue[0], 0)
        << "Grayscale pixel values should be >= 0.";
    ASSERT_LE(grayscalePixelValue[0], 255)
        << "Grayscale pixel values should be <= 255.";
}
