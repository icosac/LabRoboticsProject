//
// Created by valerio on 24/08/17.
//

#include <iostream>
#include <utils.hh>
#include <cstring>
#include <camera_capture.hh>

#define DEBUG
#ifdef DEBUG
#define SDEBUG(X) { std::cout << X << std::endl; }
#else
#define SDEBUG(X) {}
#endif


CameraCapture::CameraCapture(input_options_t options_):
    loadedCoefficients(false)
{
    SDEBUG("CameraCapture::openCamera() quered params"
                   << "\n\tnameCamera:      "<<options_.nameCamera
                   << "\n\tframe Width  px: "<<options_.frameWidth_px
                   << "\n\tframe Height px: "<<options_.frameHeight_px
                   << "\n\tcameraFPS:       "<<options_.cameraFPS);
    options = options_;

    if (!startCamera()) throw std::runtime_error("Failed to open camera " + std::string(options_.nameCamera));
    statusOk = true;

    SDEBUG("CameraCapture::openCamera() actual params"
                   << "\n\tnameCamera:      "<<options_.nameCamera
                   << "\n\tframe Width  px: "<<get(cv::CAP_PROP_FRAME_WIDTH)
                   << "\n\tframe Height px: "<<get(cv::CAP_PROP_FRAME_HEIGHT)
                   << "\n\tcameraFPS:       "<<get(cv::CAP_PROP_FPS));
}

bool CameraCapture::startCamera(){
    if(cv::VideoCapture::open( options.nameCamera)){
        /* Set up the capture device */
        return set(cv::CAP_PROP_FRAME_WIDTH,options.frameWidth_px)
               && set(cv::CAP_PROP_FRAME_HEIGHT,options.frameHeight_px)
               && set(cv::CAP_PROP_FPS,options.cameraFPS);
               //&& set(cv::CAP_PROP_GAIN, 1);
    }
    return false;
}

bool CameraCapture::grab(cv::Mat & img, double & timestamp)
{
    if(!statusOk && !startCamera()) return false;

    cv::Mat orig_img;
    timestamp = timeutils::getTimeS();
    if(!cv::VideoCapture::read(orig_img)){
        statusOk = false;
        return false;
    }
    statusOk = true;

    if (loadedCoefficients)
    {
        cv::undistort(orig_img, img, cameraMatrix, distCoeffs);
    }
    else
    {
        img = orig_img;
    }
    return statusOk;
}

CameraCapture::~CameraCapture() {

    /* Verify if the device is active */
    if (!isOpened())
        return;  // camera is already close

    /* Close the frame grabber */
    release();
    return;
}

bool CameraCapture::isOpened() {
    return cv::VideoCapture::isOpened();
}

bool CameraCapture::isAlive(){
    if(!isOpened()) return false;
    return statusOk;
}

bool CameraCapture::loadCoefficients(std::string const & filename)
{
  cv::FileStorage fs( filename, cv::FileStorage::READ );
  if (!fs.isOpened())
  {
    return false;
  }
  // TODO: add try/catch for loading coefficients
  fs["camera_matrix"] >> cameraMatrix;
  fs["distortion_coefficients"] >> distCoeffs;
  fs.release();
  loadedCoefficients = true;
//  std::cout << cameraMatrix << std::endl;
//  std::cout << distCoeffs << std::endl;
  return true;
}


CameraCapture::input_options_t::input_options_t() : frameHeight_px(240),
                                                    frameWidth_px(320), cameraFPS(5), nameCamera("/dev/video0") {}

CameraCapture::input_options_t::input_options_t(const uint32_t frameHeight_px_, const uint32_t frameWidth_px_,
                                                const uint32_t cameraFPS_, const uint32_t cameraId_):
        frameHeight_px(frameHeight_px_), frameWidth_px(frameWidth_px_), cameraFPS(cameraFPS_)
{
    /* Create the name of the qr_reader */
    sprintf(nameCamera, "/dev/video%d%c", cameraId_, 0);
}

CameraCapture::input_options_t::input_options_t(const CameraCapture::input_options_t &inpOpt) :
        frameHeight_px(inpOpt.frameHeight_px), frameWidth_px(inpOpt.frameWidth_px),
        cameraFPS(inpOpt.cameraFPS) {
    std::strcpy(nameCamera,inpOpt.nameCamera);
}



