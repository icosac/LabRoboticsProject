//
// Created by valerio on 24/08/17.
//

#ifndef CAMERACAPTURE_HH
#define CAMERACAPTURE_HH

#include <opencv2/opencv.hpp>

class CameraCapture: private cv::VideoCapture 
{

public:
    /**
   * @brief Structure for store the input option for the class CameraCapture
   *
   * frameHeight_px desidered height of the camera
   *
   * frameWidth_px desidered width of the frame of the camera
   *
   * cameraFPS desidered FPS of the camera
   *
   * nameCamera  is the camera filedescriptor (max 20 char)
   *
   */
    struct input_options_t {
        uint32_t frameHeight_px;
        uint32_t frameWidth_px;
        uint32_t cameraFPS;
        char nameCamera[20];

        // Default initializer
        input_options_t();


        // Initializer
        input_options_t(const uint32_t frameHeight_px_,const uint32_t frameWidth_px_,
                        const uint32_t cameraFPS_,const uint32_t cameraId_);

        // Copy constructor
        input_options_t(const input_options_t& inpOpt);
    };


    /**
     * Initializer of the camera capture class
     * @param options  for the class
     * @return
     */
    CameraCapture(input_options_t options);

    /**
     * Grab the first frame available and store it in frame variable
     * @return success if a frame is grabbed, false if not
     */
    bool grab(cv::Mat & img, double & timestamp);

    /**
     * Check if the videostream is opened
     * @return true if open, false if not
     */
    bool isOpened();

    /**
    * Check if the videostream is alive
    * @return true if open, false if not
    */
    bool isAlive();

    /**
     * release the resource
     */
    ~CameraCapture();

    /**
     * get time in ns
     * @return time in ns
     */
    //uint64_t get_Time_us() const;

    bool startCamera();

    bool loadCoefficients(std::string const & filename);

private:
    input_options_t options;

    bool loadedCoefficients;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    bool statusOk;

};


#endif 
