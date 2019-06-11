#ifndef UTILS_HH
#define UTILS_HH


// #include <tesseract/baseapi.h> // Tesseract headers
// #include <leptonica/allheaders.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

void my_imshow(const char* win_name, cv::Mat img, bool reset=false);


#ifdef DEBUG
#include <cstdio>
#include <iostream>
#endif
using namespace std;

void TOFILE(const char* fl_name, const char* msg);
void CLEARFILE(const char* fl_name);

#ifdef DEBUG
  #define INFO(x) cerr << #x << ": " << x << endl;
  #define INFOS(x) cerr << x << endl;
  #define INFOV(v) \
    for (auto x : v) \
      INFOS(x)
#else 
  #define INFO(x)
  #define INFOS(x)
  #define INFOV(v)
#endif


#endif //UTILS_HH
