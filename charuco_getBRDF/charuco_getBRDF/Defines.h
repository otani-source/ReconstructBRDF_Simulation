#pragma once
#include <opencv2/opencv.hpp>

cv::String getfilename(char kof);
bool readBRDFFile(cv::Mat brdf2d, cv::String);
cv::String saveFilename(char kof);
bool saveAsFile(cv::Mat data, cv::String filename);

void randomData();
int estimate();
cv::String replaceStr(std::string str, cv::String prev, cv::String next);

constexpr auto N = 90U;

int cal_angle(const cv::Vec3d a, const cv::Vec3d b);

cv::Vec3d operator*(const cv::Mat &m, const cv::Vec3d &v);