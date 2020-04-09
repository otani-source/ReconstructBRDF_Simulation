#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <Windows.h>
#include <FlyCapture2.h>
#include "PGROpenCV.h"
#include "Defines.h"
#include "Parameters.h"

using namespace std;
using namespace cv;
using Fst = FileStorage;
namespace au = aruco;
namespace Fc = FlyCapture2;

int m = 810;

static const Vec3d Scale(1. / 1500, 1.15 / 1500, 1.66 / 1500);

constexpr array<int, 3> dictId = { au::DICT_6X6_250, au::DICT_4X4_50, au::DICT_5X5_50 };

static bool readCameraParameters(String filename, Mat &camMatrix, Mat &distCoeffs) {
	Fst fs(filename, Fst::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

int estimate() {
	Mat camMatrix, distCoeffs;

	array<Ptr<au::Dictionary>, 3> dictionary;
	array<Ptr<au::CharucoBoard>, 3> charucoboard;
	array<Ptr<au::Board>, 3> board;
	Mat1d rmtx;
	Mat3d brdf2d(N, N);

	bool readOk = readCameraParameters("./camera.xml", camMatrix, distCoeffs);
	if (!readOk) {
		std::cerr << "Invalid camera file\n";
		return 1;
	}

	Ptr<au::DetectorParameters> detectorParams = au::DetectorParameters::create();

	float axisLength = 0.5f * ((float)min(squaresX[0], squaresY[0]) * (squareLength));

	for (int i = 0; i < dictionary.size(); i++) {
		dictionary[i] = au::getPredefinedDictionary(au::PREDEFINED_DICTIONARY_NAME(dictId[i]));

		// create charuco board object
		charucoboard[i] = au::CharucoBoard::create(squaresX[i], squaresY[i], squareLength, markerLength, dictionary[i]);
		//board[i] = charucoboard[i].staticCast<aruco::Board>();
	}

	double totalTime = 0;
	int totalIterations = 0;

	String filename = getfilename('b');
	
	if (!readBRDFFile(brdf2d, filename)) {
		return -1;
	}

	Mat3b out_img(N, N);
	brdf2d.convertTo(out_img, CV_8U, 255);
	cvtColor(out_img, out_img, COLOR_RGB2BGR);

	// 結果画像表示
	Mat3b getImg(N, N, cv::Vec3b(0, 0, 0));
	Mat3b getAngle(N, N, cv::Vec3b(0, 0, 0));

	VideoWriter writer("C:/Users/otani/Desktop/out.wmv", cv::VideoWriter::fourcc('W', 'M', 'V', '1'), 10, cv::Size(1640, 960));
	
	if (!writer.isOpened()) return -1;
	TPGROpenCV pgrOpenCV;

	// initialization
	pgrOpenCV.init(FlyCapture2::PIXEL_FORMAT_BGR);

	// start capturing
	if (pgrOpenCV.start() < 0)
		return 6;

	char key = 0;
	Mat tmp(N, N, CV_64FC3, Scalar(-1, -1, -1));
	Mat3d outData = tmp;
	Vec3d ivec(1, 0, 0.00000001);
	Mat1d rotI(3, 3);
	Rodrigues(Vec3d(0, -CV_PI / 12, 0), rotI);

	using chsys = chrono::system_clock;
	int64 elapsed;

	int num = 0;
	int timestack = 0;
	filename = replaceStr(filename, "material", "2dbrdf");
	filename = replaceStr(filename, ".binary", "_");
	int psu = 0;
	int su = 0;
	String times;

	auto stepst = chsys::now();
	auto start = chsys::now();

	Vec3d rvec, tvec;
	Mat tmpimage, imageCopy, image;

	while (key != 'q' && key != 27) {

		ivec = normalize(ivec);
		pgrOpenCV.queryFrame();

		vector< int > markerIds, charucoIds;
		vector< vector< cv::Point2f > > markerCorners, rejectedMarkers;
		vector< cv::Point2f > charucoCorners;

		bool detecting = false;

		for (int i = 0; i < 3; i++) {
			image = pgrOpenCV.getVideo();
			cv::resize(image, image, cv::Size(1280, 960), 0, 0, cv::INTER_NEAREST);
			double tick = (double)cv::getTickCount();

			// detect markers
			au::detectMarkers(image, dictionary[i], markerCorners, markerIds, detectorParams, rejectedMarkers);

			// interpolate charuco corners
			int interpolatedCorners = 0;
			if (markerIds.size() > 0) {
				interpolatedCorners =
					au::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard[i],
						charucoCorners, charucoIds, camMatrix, distCoeffs);
				detecting = true;
			}

			// estimate charuco board pose
			bool validPose = false;
			if (camMatrix.total() != 0)
				validPose = au::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard[i],
					camMatrix, distCoeffs, rvec, tvec);

			double currentTime = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();
			totalTime += currentTime;
			totalIterations++;
			if (totalIterations % 30 == 0) {
				std::cout << "Detection Time = " << currentTime * 1000 << " ms (Mean = "
						<< 1000 * totalTime / double(totalIterations) << " ms)\n";
			}

			// draw results
			image.copyTo(imageCopy);
			if (markerIds.size() > 0) {
				au::drawDetectedMarkers(imageCopy, markerCorners);
			}

			if (interpolatedCorners > 0) {
				au::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
			}
				
			if (validPose) {
				Rodrigues(rvec, rmtx);			// 回転ベクトルを回転行列に変換
				if ((i == 0) || (charucoIds[0] < 12)) {
					tvec += rmtx * deltaAxis[i];	// 軸を移動
					if (i > 0) {
						rmtx = rmtx * amtx[i - 1];
					}
				}
				else {
					tvec += rmtx * deltaAxis[i + 2];
					rmtx = rmtx * amtx[i + 1];
				}

				if (i > 0) {
					Rodrigues(rmtx, rvec);
				}

				
				// 軸を描画
				au::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

				Vec3d ovec = normalize(-rmtx.inv() * tvec);	// 視点ベクトル
				Vec3d hvec = normalize(ivec + ovec);		// ハーフベクトル

				int thdeg = cal_angle(Vec3d(0, 0, 1), hvec);	// theta_h(degree)
				int tddeg = cal_angle(ovec, hvec);				// theta_d(degree)

				if (thdeg < 90 && tddeg < 90) {
					std::cout << thdeg << ',' << tddeg << "\n";
					getImg(tddeg, thdeg) = out_img(tddeg, thdeg);
					outData(tddeg, thdeg) = brdf2d(tddeg, thdeg);
					getAngle(tddeg, thdeg) = Vec3d(255, 255, 255);
				}

				break;
			}
		}

		Mat tmpImg, tmpAngle;
		cv::resize(getImg, tmpImg, cv::Size(), 4, 4, cv::INTER_NEAREST);
		cv::resize(getAngle, tmpAngle, cv::Size(), 4, 4, cv::INTER_NEAREST);

		cout << tvec << endl;

		Point p = cv::Point(tvec[0], tvec[1]);

		if (detecting) {
			//cv::Point p = (cv::Point)charucoCorners[0];

			cv::circle(imageCopy, p, 20, cv::Scalar(200, 200, 200), -1, CV_AA);
		}

		Mat3b display(imageCopy.rows, imageCopy.cols + tmpImg.cols, cv::Vec3b(80, 80, 80));
		imageCopy.copyTo(display(cv::Rect(0, 0, imageCopy.cols, imageCopy.rows)));
		tmpImg.copyTo(display(cv::Rect(imageCopy.cols, 0, tmpImg.cols, tmpImg.rows)));
		tmpAngle.copyTo(display(cv::Rect(imageCopy.cols, display.rows - tmpImg.rows, tmpAngle.cols, tmpAngle.rows)));

		psu = su;
		su = (int)cv::sum(getAngle / 255)[0];

		String text = "Incident angle = " + to_string(cal_angle(ivec, Vec3d(0, 0, 1)));
		String text2 = "m = " + to_string(su) + " / " + to_string(m);
		cv::putText(display, text, Point(1300, 400), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
		cv::putText(display, text2, Point(1300, 450), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			
		auto stepend = chsys::now();
		elapsed = chrono::duration_cast<chrono::seconds>(stepend - start).count() - timestack;

		int min = (int)elapsed / 60;
		int sec = (int)elapsed % 60;
		times = to_string(min) + "m" + to_string(sec) + "s";
		cv::putText(display, "time = " + times, Point(1300, 500), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));

		cv::imshow("chart", display);
		//cout << display.size() << '\n';
		writer << Mat(display);

		key = (char)cv::waitKey(waitTime);

		elapsed = chrono::duration_cast<chrono::seconds>(stepend - stepst).count();
			
		if (elapsed >= 10) {
			ivec = rotI * ivec;
			stepst = chsys::now();
			if (cal_angle(ivec, Vec3d(0, 0, 1)) <= 0 || cal_angle(ivec, Vec3d(0, 0, 1)) >= 89) {
				rotI = rotI.inv();
			}
		}

		// 目標に達したら2DBRDFに書き出す
		if ((su == 810 || su == 1620 || su == 3240) && su - psu == 1) {
			String savename = filename + to_string(su / 81) + "p_" + times + ".brdf2";
			if (!saveAsFile(outData, savename)) {
				return -1;
			}
		}
		if (su == m) {
			if (m == 3240) break;
			m *= 2;
		}

		if (key == 'c') {
			imwrite("C:/Users/otani/GoogleDrive/sotsuron/fig/imageCopy.png", imageCopy);
			imwrite("C:/Users/otani/GoogleDrive/sotsuron/fig/display.png", display);
			std::cout << "Write image.\n";
		}
	}
	auto end = chsys::now();
	elapsed = chrono::duration_cast<chrono::seconds>(end - start).count() - timestack;
	std::cout << times << '\n';

	// stop capturing
	pgrOpenCV.stop();

	// finalization
	pgrOpenCV.release();
	return 0;
}
