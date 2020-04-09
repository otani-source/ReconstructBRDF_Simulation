#pragma once

constexpr std::array<int, 3> squaresX = { 5, 5, 6 };
constexpr std::array<int, 3> squaresY = { 6, 6, 6 };
constexpr float squareLength = 0.037f;
constexpr float markerLength = 0.022f;
constexpr int waitTime = 10;

const enum {
	f56,
	f35_a,
	f36_a,
	f35_b,
	f36_b
};

constexpr float sx = squareLength * squaresX[f56];
constexpr float sy = squareLength * squaresY[f56];

std::array<cv::Vec3d, 5> deltaAxis = {
	cv::Vec3d(sx * 0.5, sy * 0.5),
	cv::Vec3d(sx * 0.5, -0.009, -sy * 0.5 - 0.004),
	cv::Vec3d(sy * 0.5, -0.003, -sx * 0.5 - 0.006),
	cv::Vec3d(sx * 0.5, sy * 0.5 - 0.006, -sy * 0.5 - 0.005),
	cv::Vec3d(sy * 0.5, sy * 0.5 - 0.007, -sx * 0.5 - 0.007)
};

std::array<cv::Mat1d, 4> amtx = {
	(cv::Mat1d(3, 3) << -1, 0, 0, 0, 0, -1, 0, -1, 0),
	(cv::Mat1d(3, 3) << 0, 1, 0, 0, 0, -1, -1, 0, 0),
	(cv::Mat1d(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0),
	(cv::Mat1d(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0)
};