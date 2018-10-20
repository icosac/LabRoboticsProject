// the core of the project

//maybe not all the libraries are neccessary...
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

static const int SIZE     = 330;
static const int W_0      = 0;
static const int H_0      = 0;
static const int W_OFFSET = 20;
static const int H_OFFSET = 90;
static const int LIMIT    = W_0 + 4*SIZE + 3*W_OFFSET;
int W_now = W_0;
int H_now = H_0;

const string filename = "../data/map/01.jpg";
const string calib_file = "../support_file/intrinsic_calibration.xml";

void loadCoefficients(const string& filename, Mat& camera_matrix, Mat& dist_coeffs);
void my_imshow(const char*  win_name, Mat img);

int main(){
    // Load image from file
    Mat or_img = imread(filename.c_str());
    if(or_img.empty()) {
        throw runtime_error("Failed to open the file " + filename);
    }
    
    // Display original image
    my_imshow("Original", or_img);

    // fix calibration with matrix
    Mat camera_matrix, dist_coeffs;
    loadCoefficients(calib_file, camera_matrix, dist_coeffs);

    Mat fix_img;
    undistort(or_img, fix_img, camera_matrix, dist_coeffs);
    
    // Display fixed image
    my_imshow("Fixed", fix_img);

    //Convert from RGB to HSV= Hue-Saturation-Value
    Mat hsv_img;
    cvtColor(fix_img, hsv_img, COLOR_BGR2HSV);

    // Display HSV image
    my_imshow("HSVimage", hsv_img);

    // Find black regions (filter on saturation and value)
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    Mat black_mask;
    inRange(hsv_img, Scalar(0, 0, 0), Scalar(180, 255, 50), black_mask);  
    my_imshow("BLACK_filter", black_mask);

    // wait a char 'q' to proceed
    while((char)waitKey(1)!='q'){}
return(0);
}


void loadCoefficients(const string& filename, Mat& camera_matrix, Mat& dist_coeffs){
  FileStorage fs(filename, FileStorage::READ );
  if (!fs.isOpened()){
    throw runtime_error("Could not open file " + filename);
  }
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> dist_coeffs;
  fs.release();
}

void my_imshow(const char*  win_name, Mat img){
    //const string s = win_name;
    namedWindow(win_name, CV_WINDOW_NORMAL);
    cvvResizeWindow(win_name, SIZE, SIZE);
    imshow(win_name, img);
    moveWindow(win_name, W_now, H_now);
    //cout << W_now << " " << H_now << endl;
    W_now += SIZE + W_OFFSET;
    if(W_now >= LIMIT){
        W_now = W_0;
        H_now += SIZE + H_OFFSET;
    }
}