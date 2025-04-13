#ifndef __FOURIER_MELLIN_SIMPLE_H__
#define __FOURIER_MELLIN_SIMPLE_H__

#include <iostream>
#include <string_view>
#include <tuple>

#include "transform.hpp"
#include "utilities.hpp"

class FourierMellinSimple {
   public:
    FourierMellinSimple(std::string_view reference_fp);

    Transform RegisterImage(const cv::Mat& target) const;
    Transform RegisterImage(std::string_view target_fp) const;

    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat& target) const;
    std::tuple<cv::Mat, Transform> GetRegisteredImage(std::string_view target_fp) const;

    FourierMellinSimple(const FourierMellinSimple&) = delete;
    FourierMellinSimple(FourierMellinSimple&&) = delete;
    FourierMellinSimple& operator=(const FourierMellinSimple&) = delete;
    FourierMellinSimple& operator=(FourierMellinSimple&&) = delete;
    ~FourierMellinSimple() = default;

   private:
    cv::Mat ConvertImageToLogPolar(const cv::Mat& img) const;

   private:
    int width_, height_;
    LogPolarMap logPolarMap_;

    cv::Mat referenceImg_;
    cv::Mat referenceImgLogPolar_;
};

#endif  // __FOURIER_MELLIN_SIMPLE_H__