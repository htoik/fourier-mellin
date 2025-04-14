#ifndef __LOG_POLAR_MAP_H__
#define __LOG_POLAR_MAP_H__

#include <opencv2/core/mat.hpp>

class LogPolarMap {
   public:
   LogPolarMap();
   LogPolarMap(int width, int height);
    
    cv::Mat ConvertToLogPolar(const cv::Mat& img) const;
    
    int GetLogPolarSize() const;
    double GetLogBase() const;

    ~LogPolarMap() = default;
    LogPolarMap(const LogPolarMap&) = delete;
    LogPolarMap(LogPolarMap&& rhs) = delete;
    LogPolarMap& operator=(const LogPolarMap&) = delete;
    LogPolarMap& operator=(LogPolarMap&&rhs) = delete;

   private:
    void ConstructMaps();

   private:
    int width_, height_;
    int logPolarSize_;
    double logBase_;
    cv::Mat xMap_;
    cv::Mat yMap_;
};

#endif  // __LOG_POLAR_MAP_H__