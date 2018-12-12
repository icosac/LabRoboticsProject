#ifndef UTILS_HH
#define UTILS_HH


// #include <tesseract/baseapi.h> // Tesseract headers
// #include <leptonica/allheaders.h>

// #include <opencv2/highgui.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>

#ifdef DEBUG
#include <cstdio>
#include <iostream>
#endif

// using namespace cv;
using namespace std;

// void my_imshow(const char* win_name, Mat img, bool reset=false);

void TOFILE(const char* fl_name, const char* msg);
void CLEARFILE(const char* fl_name);

#ifdef DEBUG
  #define INFO(x) cerr << #x << ": " << x << endl;
  #define INFOS(x) cerr << x << endl;
  #define INFOV(v) \
    for (auto x : v) \
      INFO(x)
#else 
  #define INFO(x)
  #define INFOS(x)
  #define INFOV(v)
#endif


#endif //UTILS_HH
