#include "PGROpenCV.h"

// Constructor
TPGROpenCV::TPGROpenCV()
{

}

// Destructor
TPGROpenCV::~TPGROpenCV()
{

}

// initialize PGR
int TPGROpenCV::init( FlyCapture2::PixelFormat _format )
{
	// get number of cameras on the bus
	fc2Error = fc2BusMgr.GetNumOfCameras( &numCameras );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	} else {
		std::cout << "Number of cameras detected: " << numCameras << std::endl;
	}

	// get guid from index
	fc2Error = fc2BusMgr.GetCameraFromIndex( 0, &fc2Guid );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	}

	// connect to the camera using guid
	fc2Error = fc2Cam.Connect( &fc2Guid );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	}

	// get the camera information
	fc2Error = fc2Cam.GetCameraInfo( &fc2CamInfo );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	} else {
		PrintCameraInfo( &fc2CamInfo );
	}

	// set the pixel format
	fc2PixelFormat = _format;
}

void TPGROpenCV::PrintBuildInfo()
{
	FlyCapture2::FC2Version fc2Version;
	FlyCapture2::Utilities::GetLibraryVersion( &fc2Version );

	std::cout << "FlyCapture2 library version: " << fc2Version.major << "." << fc2Version.minor << "." << fc2Version.type << "." << fc2Version.build << std::endl;
	std::cout << "Application build date: " << __DATE__ << " " << __TIME__ << std::endl << std::endl;
}

void TPGROpenCV::PrintError( FlyCapture2::Error error )
{
	error.PrintErrorTrace();
}

void TPGROpenCV::PrintCameraInfo( FlyCapture2::CameraInfo* pCamInfo )
{
	std::cout << std::endl	<< "\n*** CAMERA INFORMATION ***" << std::endl
							<< "Serial number - " << pCamInfo->serialNumber << std::endl
							<< "Camera model - " << pCamInfo->modelName << std::endl
							<< "Camera vendor - " << pCamInfo->vendorName << std::endl
							<< "Sensor - " << pCamInfo->sensorInfo << std::endl
							<< "Resolution - " << pCamInfo->sensorResolution << std::endl
							<< "Firmware version - " << pCamInfo->firmwareVersion << std::endl
							<< "Firmware build time - " << pCamInfo->firmwareBuildTime << std::endl << std::endl;
}

// reply present pixel format in OpenCV style
int TPGROpenCV::PixelFormatInOpenCV()
{
	switch( fc2PixelFormat ) {
	case FlyCapture2::PIXEL_FORMAT_BGR:
	case FlyCapture2::PIXEL_FORMAT_BGRU:
	case FlyCapture2::PIXEL_FORMAT_RGB:
	case FlyCapture2::PIXEL_FORMAT_RGBU:
		return CV_8UC3;
		break;
	case FlyCapture2::PIXEL_FORMAT_S_RGB16:
		return CV_16SC3;
		break;
	case FlyCapture2::PIXEL_FORMAT_BGR16:
	case FlyCapture2::PIXEL_FORMAT_BGRU16:
	case FlyCapture2::PIXEL_FORMAT_RGB16:
		return CV_16UC3;
		break;
	case FlyCapture2::PIXEL_FORMAT_MONO8:
	case FlyCapture2::PIXEL_FORMAT_RAW8:
		return CV_8UC1;
		break;
	case FlyCapture2::PIXEL_FORMAT_MONO16:
	case FlyCapture2::PIXEL_FORMAT_RAW16:
		return CV_16UC1;
		break;
	case FlyCapture2::PIXEL_FORMAT_S_MONO16:
		return CV_16SC1;
		break;
	default:
		return CV_8UC3;
		break;
	}
}

int TPGROpenCV::start()
{
	fc2Error = fc2Cam.StartCapture();
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	} else {
		// retrieve an image to allocate fc2Mat
		FlyCapture2::Image	wk;
		fc2Error = fc2Cam.RetrieveBuffer( &wk );
		if( fc2Error != FlyCapture2::PGRERROR_OK ) {
			PrintError( fc2Error );
			return -1;
		} else {
			fc2Mat.create( wk.GetRows(), wk.GetCols(), PixelFormatInOpenCV() );
		}
		return 0;
	}
}

int TPGROpenCV::queryFrame()
{
	// retrieve a frame image
	FlyCapture2::Image	rawImage;
	fc2Error = fc2Cam.RetrieveBuffer( &rawImage );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	}

	// convert the raw image
	FlyCapture2::Image	cvtImage;
	fc2Error = rawImage.Convert( fc2PixelFormat, &cvtImage );
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	}

	memcpy( fc2Mat.data, cvtImage.GetData(), cvtImage.GetDataSize() );

	return 0;
}

int TPGROpenCV::stop()
{
	fc2Error = fc2Cam.StopCapture();
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	} else {
		return 0;
	}
}

int TPGROpenCV::release()
{
	fc2Mat.release();

	fc2Error = fc2Cam.Disconnect();
	if( fc2Error != FlyCapture2::PGRERROR_OK ) {
		PrintError( fc2Error );
		return -1;
	} else {
		return 0;
	}
}
