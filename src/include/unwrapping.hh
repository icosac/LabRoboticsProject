#ifndef UNWRAPPING_HH
#define UNWRAPPING_HH

#include <utils.hh>
#include <settings.hh>
#include <detection.hh>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

/*! \brief Take some images according to a xml and unwrap the black rectangle inside the image after appling undistortion trasformation.
    \details Load from the xml file 'data/settings.xml' the name of some images, load the images from the file,\n
    apply the calibration (undistortion trasformation) thanks to the matrices load with the 'loadCoefficients' function.\n
    Then, with the use of a filter for the black the region of interest (a rectangle) is identified and all the perspective is rotated for reach a top view of the rectangle.\n
    Finally, the images are saved on some files.

    \param[in] _imgRead Boolean flag that says if load or not the image from file, or as a function parameter. In addition, also the return procedure change if true the image is saved on the disk otherwise is saved on the img function parameter. True=load and store on file.
    \param[in/out] img The image that eventually is loaded from the function. And the one that will be modified for returning the elaborated frame.
    \returns A 0 is return if the function reach the end.
*/
int unwrapping(const bool _imgRead=true, Mat * img=nullptr);

/*! \brief Store in the given vector the white corners in the same order as the given black ones.
    \param[in] rectLow A vector where the low corners of the rectangle (black markers position) are stored.
    \param[out] rectHigh A vector where the high corners of the rectangle (white markers position) will be stored.
*/
void createPointsHigh(const vector<Point> & rectLow, vector<Point> & rectHigh);

/*! \brief Load coefficients from a file.
    \details Load two matrix 'camera_matrix' and 'distortion_coefficients' from the xml file passed.
    \param[in] filename The string that identify the location of the xml file.
    \param[out] camera_matrix Where the 'camera_matrix' matrix is saved.
    \param[out] dist_coeffs Where the 'distortion_coefficients' matrix is saved.
*/
void loadCoefficients(  const string filename,
                        Mat& camera_matrix, 
                        Mat& dist_coeffs);

/*! \brief Since the border of the arena might not always be clean but might have some imperfection, this functions computes the four vertixes taking all the points and computing the four that are the clostest to the corner of the image.
	\param[in] _rect The vector of cv::Point to work on.
	\param[in] width The width of the image.
	\param[in] height The height of the image.
*/
void find_rect(	vector<Point>& _rect, 
								const int& width,
								const int& height);

#endif