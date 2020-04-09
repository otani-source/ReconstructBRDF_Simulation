#pragma once

#include <FlyCapture2.h>
#include <opencv2\opencv.hpp>
#include <cstdio>
#include <iostream>

class TPGROpenCV
{
private:
	FlyCapture2::Error			fc2Error;
	FlyCapture2::BusManager		fc2BusMgr;
	FlyCapture2::PGRGuid		fc2Guid;
	FlyCapture2::CameraInfo		fc2CamInfo;
	FlyCapture2::Camera			fc2Cam;
	FlyCapture2::PixelFormat	fc2PixelFormat;
	unsigned int				numCameras;

	cv::Mat						fc2Mat;
public:
	TPGROpenCV();
	~TPGROpenCV();
	int init( FlyCapture2::PixelFormat _format = FlyCapture2::PIXEL_FORMAT_BGR );
	void PrintBuildInfo();
	void PrintError( FlyCapture2::Error error );
	void PrintCameraInfo( FlyCapture2::CameraInfo* pCamInfo );
	int PixelFormatInOpenCV();
	int start();
	int queryFrame();
	int stop();
	int release();
	cv::Mat getVideo(){ return fc2Mat; };
};

