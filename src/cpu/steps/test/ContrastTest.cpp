#include "ContrastStep.h"
#include "Image.h"
#include "gtest/gtest.h"

TEST(ContrastTest, AdjustsContrastCorrectly)
{
    // Arrange
    cv::Mat originalImage = cv::Mat::ones(100, 100, CV_8UC3) *
                            128;  // Create a 100x100 image with mid-level gray
    Image img(originalImage);     // Wrap in the Image class
    ContrastStep contrastStep(1.5f);  // Increase contrast by 50%

    // Act
    contrastStep.process(img);
    cv::Mat contrastAdjustedImage = img.getImage();

    // Assert
    // Check if the pixel values are properly adjusted for contrast
    // The contrast should scale the pixel values around the mean (128 in this
    // case)
    ASSERT_GT(contrastAdjustedImage.at<cv::Vec3b>(0, 0)[0],
              128);  // Brightness should increase for higher values
    ASSERT_LT(contrastAdjustedImage.at<cv::Vec3b>(0, 0)[0],
              255);  // Should not exceed the maximum value (255)
}
