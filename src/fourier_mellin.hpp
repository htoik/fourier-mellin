#ifndef __FOURIER_MELLIN_H__
#define __FOURIER_MELLIN_H__

#include <iostream>
#include <string_view>
#include <tuple>

#include "image_filter.hpp"
#include "log_polar_map.hpp"
#include "transform.hpp"
#include "utilities.hpp"

class FourierMellin {
public:
  FourierMellin(const cv::Mat &reference);
  FourierMellin(std::string_view reference_fp);

  Transform RegisterImage(const cv::Mat &target) const;
  Transform RegisterImage(std::string_view target_fp) const;

  std::tuple<cv::Mat, Transform>
  GetRegisteredImage(const cv::Mat &target) const;
  std::tuple<cv::Mat, Transform>
  GetRegisteredImage(std::string_view target_fp) const;

  FourierMellin(const FourierMellin &) = delete;
  FourierMellin(FourierMellin &&) = delete;
  FourierMellin &operator=(const FourierMellin &) = delete;
  FourierMellin &operator=(FourierMellin &&) = delete;
  ~FourierMellin() = default;

private:
  cv::Mat ReadGrayscaleImageFromFile(std::string_view img_fp) const;
  cv::Mat ConvertImageToLogPolar(const cv::Mat &img) const;

private:
  int width_, height_;
  LogPolarMap logPolarMap_;
  ImageFilter imageFilter_;
  // cv::Mat highPassFilter_;
  // cv::Mat apodizationWindow_;

  cv::Mat referenceImg_;
  cv::Mat referenceImgLogPolar_;
};

#endif // __FOURIER_MELLIN_H__