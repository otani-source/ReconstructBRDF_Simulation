#include <opencv2/opencv.hpp>
#include <fstream>
#include <Windows.h>
#include <string>
#include "Defines.h"

using namespace std;
using namespace cv;

static const Vec3d Scale(1. / 1500, 1.15 / 1500, 1.66 / 1500);

int maind()
{
	Mat3d brdf2d(N*N, 1);

	

	brdf2d = brdf2d.reshape(3, N).t();
	Mat3d cimg(N, N);

	double theta_h;
	int idx;
	for (int i = 0; i < N; i++) {
		theta_h = 0.5 * CV_PI * i / N;
		idx = (int)(N*sqrt(2*theta_h / CV_PI));
		cout << idx << endl;
		brdf2d.col(idx).copyTo(cimg.col(i));
	}

	Mat3b out_img(N, N);
	cimg.convertTo(out_img, CV_8U, 255);
	cvtColor(out_img, out_img, COLOR_RGB2BGR);

	namedWindow("BRDF", WINDOW_KEEPRATIO);
	imshow("BRDF", out_img);

	waitKey(0);

	return 0;
}

int cal_angle(const Vec3d a, const Vec3d b) {
	return (int)(180 * acos(a.dot(b)) / CV_PI);
}

Vec3d operator*(const Mat &m, const Vec3d &v) {
	return (Vec3d)Mat(m * Mat(v));
}

String getfilename(char kof) {
	OPENFILENAME ofn;
	char szFile[MAX_PATH + 1] = "";

	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);

	switch (kof) {
	case 'v':
		ofn.lpstrFilter =
			"movファイル(*.mov)\0*.mov\0"
			// "mp4ファイル(*.mp4)\0*.mp4\0"
			"すべてのファイル(*.*)\0*.*\0\0";
		break;
	case 'b':
		ofn.lpstrFilter =
			"BRDFバイナリ(*.binary)\0*.binary\0"
			"すべてのファイル(*.*)\0*.*\0\0";
		break;
	default:
		cerr << "Error: Please set args 'v' or 'b'.\n";
		exit(0);
	}

	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST;

	GetOpenFileName(&ofn);

	return szFile;
}

cv::String replaceStr(std::string str, cv::String prev, cv::String next) {
	auto pos = str.find(prev);  // 検索文字列が見つかった位置 （pos == 1）
	auto len = prev.length(); // 検索文字列の長さ （len == 2）
	if (pos != cv::String::npos) {
		str.replace(pos, len, next); // s == "a|b"
	}
	return str;
}

bool readBRDFFile(Mat brdf2d, String fname) {
	try {
		ifstream ifs(fname, ios::in | ios::binary);

		if (!ifs) {
			throw false;
		}

		unsigned int dims[3];
		ifs.read((char*)dims, sizeof(dims));

		auto n = dims[0] * dims[1] * dims[2];

		Mat1d brdf(3, n);
		//MatrixXd brdf(n, 3);
		ifs.read((char*)brdf.data, 3 * n * sizeof(double));

		ifs.close();

		brdf = brdf.t();
		brdf = brdf.reshape(1, n);

		for (int i = 0; i < 3; i++) {
			brdf.col(i) *= Scale[i];
		}

		// cout << brdf.row(0) << '\n';

		int count;
		Vec3d sum;

		Mat3d cimg(N*N, 1);
		for (int i = 0; i < N*N; i++) {
			count = 0;
			sum = 0;
			for (int j = 0; j < 2 * N; j++) {
				if (brdf(2 * i*N + j, 0) >= 0) {
					count++;
					sum += Vec3d(brdf.row(2 * i*N + j));
				}
			}
			cimg(i) = sum / count;
		}

		cimg = cimg.reshape(3, N).t();

		double theta_h;
		int idx;
		for (int i = 0; i < N; i++) {
			theta_h = 0.5 * CV_PI * i / N;
			idx = (int)(N*sqrt(2 * theta_h / CV_PI));
			cimg.col(idx).copyTo(brdf2d.col(i));
		}
	}
	catch (bool e) {
		cerr << "Cannot open file\n";
		return e;
	}

	return true;
}

String saveFilename(char kof) {
	static OPENFILENAME ofn;
	static char szPath[MAX_PATH];
	static char szFile[MAX_PATH] = "s_m";

	if (szPath[0] == TEXT('\0')) {
		GetCurrentDirectory(MAX_PATH, szPath);
	}
	if (ofn.lStructSize == 0) {
		ofn.lStructSize = sizeof(OPENFILENAME);
		//ofn.hwndOwner = hWnd;
		ofn.lpstrInitialDir = szPath;       // 初期フォルダ位置
		ofn.lpstrFile = szFile;       // 選択ファイル格納
		ofn.nMaxFile = MAX_PATH;
		switch (kof) {
		case 'i':
			ofn.lpstrDefExt = TEXT(".png");
			ofn.lpstrFilter =
				"pngファイル(*.png)\0*.png\0"
				"すべてのファイル(*.*)\0*.*\0\0";
			break;
		case 'b':
			ofn.lpstrDefExt = TEXT(".brdf2");
			ofn.lpstrFilter =
				"2変数BRDFバイナリ(*.brdf2)\0*.brdf2\0"
				"すべてのファイル(*.*)\0*.*\0\0";
			break;
		default:
			cerr << "Error: Please set args 'v' or 'b'.\n";
			exit(-1);
		}
		ofn.Flags = OFN_FILEMUSTEXIST | OFN_OVERWRITEPROMPT;
	}
	if (!GetSaveFileName(&ofn)) {
		exit(-1);
		//MessageBox(hWnd, szFile, TEXT("ファイル名を付けて保存"), MB_OK);
	}

	return szFile;
}

bool saveAsFile(Mat data, String filename) {
	try {
		ofstream ofs(filename, ios::out | ios::binary);

		if (!ofs) {
			throw false;
		}

		ofs.write((char*)data.data, 3 * N*N * sizeof(double));

		ofs.close();
	}
	catch (bool e) {
		cerr << "Erroe: Saving data failed.\n";
		return e;
	}

	return true;
}