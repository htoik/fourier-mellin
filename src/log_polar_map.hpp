#ifndef __LOG_POLAR_MAP_H__
#define __LOG_POLAR_MAP_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

class LogPolarMap {
   public:
   LogPolarMap();
   LogPolarMap(int width, int height);
   LogPolarMap(LogPolarMap&& rhs);
   LogPolarMap& operator=(LogPolarMap&&rhs);
    
    cv::Mat ConvertToLogPolar(const cv::Mat& img) const;
    // int GetLogPolarSize() const;
    // double GetLogBase() const;
    // std::tuple<const cv::Mat&, const cv::Mat&> GetMaps() const;

    ~LogPolarMap() = default;
    LogPolarMap(const LogPolarMap&) = delete;
    LogPolarMap& operator=(const LogPolarMap&) = delete;

   private:
    void ConstructMaps();

   public: // TODO
    int width_, height_;
    int logPolarSize_;
    double logBase_;
    cv::Mat xMap_;
    cv::Mat yMap_;
};

#endif  // __LOG_POLAR_MAP_H__