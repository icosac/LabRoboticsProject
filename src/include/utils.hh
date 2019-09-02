#ifndef UTILS_HH
#define UTILS_HH

#ifdef TESS
#include <tesseract/baseapi.h> // Tesseract headers
#include <leptonica/allheaders.h>
#endif

#include <maths.hh>

#include <iostream>


#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

//debug blocks most things, wait only something
// #define WAIT
// #define DEBUG

#define NAME(x) #x ///<Returns the name of the variable

#ifdef DEBUG
  #define COUT(x) cout << #x << ": " << x << endl; ///<Print the name of a variable and its content. Only if DEBUG is defined.
  #define INFO(msg) cout << msg << endl; ///<Print a messag to stdout
  #define INFOV(v)\
    for (auto el : v){ cout << el << ", " ; } cout << endl;
#else
  #define COUT(x) ///<Print a messag to stderr
  #define INFO(msg)  ///<Print the name of a variable and its content. Only if DEBUG is defined.
#endif

/*! \brief Function to show images in an order grill.
 * @param win_name The name of the window to use.
 * @param img The Mat containing the image.
 * @param reset If true the image is going to be placed in 0,0 i.e. the top left corner of the screen.
 */
void my_imshow(const char* win_name, Mat img, bool reset=false);

// Mat pixToMat(Pix* pix);

/*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed.
 *
 */
void mywaitkey();

/*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed. When a key is pressed a specific window is closed.
 *
 * @param windowName The window to close after pressing a key.
 */
void mywaitkey(string windowName);

/*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed. When a key is pressed some windows are closed.
 *
 * @param windowNames The names of the windows to close after pressing a key.
 */
void mywaitkey(Tuple<string> windowNames);






//Taken from Paolo Bevilaqua and Valerio Magnago
#include <time.h>
#include <cstdint>

namespace timeutils {

  int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p);
  double getTimeS();

}

#endif
