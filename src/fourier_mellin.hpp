#ifndef __FOURIER_MELLIN_H__
#define __FOURIER_MELLIN_H__

#include <iostream>
#include <map>

#include "utilities.hpp"
#include "transform.hpp"

class FourierMellin{
public:
    FourierMellin(int cols, int rows);
    ~FourierMellin();

    cv::Mat GetProcessImage(const cv::Mat &img) const;
    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1) const;

private:
    int cols_, rows_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;
};

#endif // __FOURIER_MELLIN_H__