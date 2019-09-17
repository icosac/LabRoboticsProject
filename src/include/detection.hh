#ifndef DETECTION_HH
#define DETECTION_HH

#include <utils.hh>
#include <settings.hh>
#include <filter.hh>
#include <configure.hh>
#include <unwrapping.hh>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;


// 0 -> Red
// 1 -> Green
// 2 -> Blue
// 3 -> Cyan (robot)
// 4 -> Black
enum COLOR_TYPE {RED, GREEN, BLUE, CYAN, BLACK};


/*! \brief Loads some images and detects shapes according to different colors.

    \param[in] _imgRead Boolean flag that says if load or not the image from file or as a function parameter. True=load from file.
    \param[in] img The imgage that eventually is loaded from the function.
    \returns Return 0 if the function reach the end.
*/
int detection(const bool _imgRead=true, const Mat * img=nullptr);

/*! \brief The function simply store the value of the given matrix and allow the access to it from different function location. 
    \details The transformation matrix are computed in the unwrapping phase and taken from the localization.

    \param[in] transf It is the matrix that can be stored but also retrieved.
    \param[in] get It is the flag that says if the given matrix need to be stored or retrieved.
*/
void getConversionParameters(Mat & transf, const bool get=true);

/*! \brief Identify the loation of the robot by acquiring the image from the default camera of the environment.

    \returns The configuration of the robot in this exactly moment.
*/
// Configuration2<double> localize();

/*! \brief Identify the location of the robot respect to the given image.

    \param[in] img It is the image where the robot need to be located.
    \param[in] raw It is a boolean flag that says if the img is raw and need filters or not.
    \returns Configuration of the robot in this exactly moment, according to the image.
*/
Configuration2<double> localize(const Mat & img, const bool raw=true);

/*! \brief Detect shapes inside the image according to the variable 'color'.

    \param[in] img Image on which the research will done.
    \param[in] color It is the type of reference color.
    These color identify the possible spectrum that the function search on the image.
*/
void shape_detection(const Mat & img, const COLOR_TYPE color);

/*! \brief It apply some filtering function for isolate the subject and remove the noise.
    \details An example of the sub functions called are: GaussianBlur, Erosion, Dilation and Threshold.

    \param[in, out] img Is the image on which the function apply the filtering.
    \param[in] color It is the type of reference color. According to the color the filtering functions apply can change in the type and in the order.
*/
void erode_dilation(Mat & img, const COLOR_TYPE color);

/*! \brief Given an image, in black/white format, identify all the borders that delimit the shapes.

    \param[in] img Is an image in HSV format at the base of the elaboration process.
    \param[out] original It is the original source of 'img', it is used for showing the detected contours.
    \param[in] color It is the type of reference color.
*/
void find_contours( const Mat & img,
                    const Mat & original, 
                    const COLOR_TYPE color);

/*! \brief Detect a number on an image inside a region of interest.

    \param[in] blob Identify the region of interest inside the image 'base'.
    \param[in] base Is the image where the function will going to search the number.

    \returns The number recognise, '-1' otherwise.
*/
int number_recognition(Rect blob, const Mat & base);

/*! \brief Given some vector save it in a xml file.

    \param[in] contours Is a vector that is saved in a xml file.
    \param[in] color It is the type of reference color, according to which the function decide if saved ('color==GREEN') or not ('otherwise') the vector 'victims'.
*/
void save_convex_hull(  const vector<vector<Point>> & contours,
                        const COLOR_TYPE color);

/*! \brief Load some templates and save them in the global variable 'templates'.
*/
void load_number_template();

/*! \brief Given an image identify the region of interest(ROI) and crop it out.

    \param[in,out] ROI Is the image that the function will going to elaborate.
*/
void crop_number_section(Mat & processROI);

#endif