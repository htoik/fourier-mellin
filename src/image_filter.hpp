#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <opencv2/core/mat.hpp>

class ImageFilter {
public:
  ImageFilter(int width, int height);

  cv::Mat GetFilteredImage(const cv::Mat &img) const;

  ~ImageFilter() = default;
  ImageFilter(const ImageFilter &) = delete;
  ImageFilter(ImageFilter &&) = delete;
  ImageFilter &operator=(const ImageFilter &) = delete;
  ImageFilter &operator=(ImageFilter &&) = delete;

private:
  int width_, height_;
  cv::Mat highPassFilter_;
  cv::Mat apodizationWindow_;
};

#endif // __IMAGE_FILTER_H__