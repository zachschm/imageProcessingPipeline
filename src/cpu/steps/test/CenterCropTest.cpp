#include "CenterCropStep.h"
#include "Image.h"
#include "gtest/gtest.h"

TEST(CenterCropTest, CenterCropsCorrectly)
{
    // Arrange
    cv::Mat originalImage = cv::Mat::zeros(
        200, 300, CV_8UC3);    // Create a black image with size 200x300
    Image img(originalImage);  // Wrap it in the Image class
    CenterCropStep cropStep(100, 100);  // Crop size 100x100

    // Act
    cropStep.process(img);
    cv::Mat croppedImage = img.getImage();

    // Assert
    ASSERT_EQ(croppedImage.rows, 100);  // Check if height is cropped to 100
    ASSERT_EQ(croppedImage.cols, 100);  // Check if width is cropped to 100
    ASSERT_EQ(croppedImage.type(),
              originalImage.type());  // Ensure the image type remains the same
}
