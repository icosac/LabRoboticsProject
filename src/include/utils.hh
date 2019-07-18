#ifndef UTILS_HH
#define UTILS_HH

#include <tesseract/baseapi.h> // Tesseract headers
#include <leptonica/allheaders.h>

#include <cstdio>

#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

#ifdef DEBUG
  #define INFO(msg) \
    fprintf(stderr, "%s\n", msg);
#else
  #define INFO(msg) 
#endif

void my_imshow(const char* win_name, Mat img, bool reset=false);
// Mat pixToMat(Pix* pix);
void mywaitkey();






//Taken from Paolo Bevilaqua and Valerio Magnago
#include <time.h>
#include <cstdint>

namespace timeutils {

  int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p);
  double getTimeS();

}

#endif
