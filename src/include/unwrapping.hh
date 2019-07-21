#ifndef UNWRAPPING_HH
#define UNWRAPPING_HH

#include <utils.hh>
#include <settings.hh>

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

    \returns A 0 is return if the function reach the end.
*/
int unwrapping();

/*! \brief Load coefficients from a file.
    \details Load two matrix 'camera_matrix' and 'distortion_coefficients' from the xml file passed.
    \param[in] filename The string that identify the location of the xml file.
    \param[out] camera_matrix Where the 'camera_matrix' matrix is saved.
    \param[out] dist_coeffs Where the 'distortion_coefficients' matrix is saved.
*/
void loadCoefficients(  const string filename,
                        Mat& camera_matrix, 
                        Mat& dist_coeffs);

#endif