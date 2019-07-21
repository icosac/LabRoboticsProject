#ifndef DETECTION_HH
#define DETECTION_HH

//tesseract
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <utils.hh>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

/*! \brief Loads some images and detects shapes according to different colors.

    \returns Return 0 if the function reach the end.
*/
int detection();

/*! \brief Detect shapes inside the image according to the variable 'color'.

    \param[in] img Image on which the research will done.
    \param[in] color Can has 3 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    These color identify the possible spectrum that the function search on the image.
*/
void shape_detection(const Mat & img, const int color, const Mat& un_img); //color: 0=red, 1=green, 2=blue, 3=black

/*! \brief It apply some filtering function for isolate the subject and remove the noise.
    \details An example of the sub functions called are: GaussianBlur, Erosion, Dilation and Threshold.

    \param[in, out] img Is the image on which the function apply the filtering.
    \param[in] color Can has 4 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    3 -> Black\n
    According to the color the filtering functions apply can change in the type and in the order.
*/
void erode_dilation(Mat & img, const int color);

/*! \brief Given an image, in black/white format, identify all the borders that delimit the shapes.

    \param[in] img Is an image in HSV format at the base of the elaboration process.
    \param[out] original Is the original source of 'img', it is used for showing the detected contours.
    \param[in] color Can has 3 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    Is used for decid which procedure apply to the image.
*/
void find_contours( const Mat & img,
                    Mat original, 
                    const int color);

/*! \brief Detect a number on an image inside a region of interest.

    \param[in] blob Identify the region of interest inside the image 'base'.
    \param[in] base Is the image where the function will going to search the number.

    \returns The number recognise, '-1' otherwise.
*/
int number_recognition(Rect blob, const Mat & base);

/*! \brief Given some vector save it in a xml file.

    \param[in] contours Is a vector that is saved in a xml file.
    \param[in] color Is the parameter according to which the function decide if saved ('color==1') or not ('otherwise') the vector 'victims'.
    \param[in] victims Is a vector that is saved in a xml file.
*/
void save_convex_hull(  const vector<vector<Point>> & contours,
                        const int color, 
                        const vector<int> & victims);

/*! \brief Load some templates and save them in the global variable 'templates'.
*/
void load_number_template();

/*! \brief Given an image identify the region of interest(ROI) and crop it out.

    \param[in,out] ROI Is the image that the function will going to elaborate.
*/
void crop_number_section(Mat & processROI);

#endif