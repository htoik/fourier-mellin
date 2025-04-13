#include "fourier_mellin_simple.hpp"

FourierMellin::FourierMellin(const cv::Mat& reference) : referenceImg_(reference.clone())
{
    // TODO: assumes one channel, and normalized float, like other constructor
    // if (referenceImg_.depth() != CV_32F) {
    //     referenceImg_.convertTo(referenceImg_, CV_32F, 1.0 / 255.0);
    // }

    width_ = referenceImg_.size().width;
    height_ = referenceImg_.size().height;
    logPolarMap_ = createLogPolarMap(width_, height_);
    highPassFilter_ = getHighPassFilter(height_, width_);
    apodizationWindow_ = getApodizationWindow(width_, height_, std::min(width_, height_));

    referenceImgLogPolar_ = ConvertImageToLogPolar(referenceImg_);
}

FourierMellin::FourierMellin(std::string_view reference_fp) : FourierMellin(ReadGrayscaleImageFromFile(reference_fp)) {}

Transform FourierMellin::RegisterImage(const cv::Mat& target) const {
    const auto& logPolarTarget = ConvertImageToLogPolar(target);

    double responseScaleRotation;
    auto [logScale, logRotation] = cv::phaseCorrelate(logPolarTarget, referenceImgLogPolar_, cv::noArray(), &responseScaleRotation);
    double rotation = -logRotation / logPolarMap_.logPolarSize * 180.0;
    double scale = std::pow(logPolarMap_.logBase, logScale);

    const auto center = cv::Point(width_, height_) / 2.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotation, scale);
    cv::Mat rotatedReference;
    cv::warpAffine(referenceImg_, rotatedReference, rotationMatrix, referenceImg_.size());

    double response;
    auto [xOffset, yOffset] = cv::phaseCorrelate(target, rotatedReference, cv::noArray(), &response);

    return Transform{-xOffset, yOffset, scale, rotation, response};
}

Transform FourierMellin::RegisterImage(std::string_view target_fp) const {
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return RegisterImage(target);
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(const cv::Mat& target) const {
    const auto& transform = RegisterImage(target);
    const auto& aligned = getTransformed(target, transform);
    return {aligned, transform};
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(std::string_view target_fp) const {
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return GetRegisteredImage(target);
}

cv::Mat FourierMellin::ConvertImageToLogPolar(const cv::Mat& img) const {
    auto img2 = PreprocessImage(img);
    cv::Mat log_polar;
    cv::remap(img2, log_polar, logPolarMap_.xMap, logPolarMap_.yMap, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar());
    return log_polar;
}

cv::Mat FourierMellin::PreprocessImage(const cv::Mat& img) const {
    cv::Mat apodized = img.mul(apodizationWindow_);
    cv::Mat dftResult = fft(apodized);
    cv::Mat filtered = fftShift(dftResult);
    cv::multiply(filtered, highPassFilter_, filtered);
    
    std::vector<cv::Mat> channels(2);
    cv::split(filtered, channels);
    cv::Mat magnitude;
    cv::magnitude(channels[0], channels[1], magnitude);
    return magnitude;
}

cv::Mat FourierMellin::ReadGrayscaleImageFromFile(std::string_view img_fp) const {
    cv::Mat img = cv::imread(std::string(img_fp), cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    return img;
}
