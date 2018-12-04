#ifndef UTILS_HH
#define UTILS_HH


// #include <tesseract/baseapi.h> // Tesseract headers
// #include <leptonica/allheaders.h>

#include <cstdio>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

void TOFILE(const char* fl_name, const char* msg);
void CLEARFILE(const char* fl_name);

#ifdef DEBUG
  #define INFO(msg) \
    fprintf(stderr, "%s\n", msg);

  void TOFILE(const char* fl_name, const char* msg) {
    FILE* fl=fopen(fl_name, "a");
    fprintf(fl, "%s", msg);
    fclose(fl);
	}
  void CLEARFILE(const char* fl_name){
    FILE* fl1=fopen(fl_name, "w");
    fclose(fl1);
	}

#else
  #define INFO(msg) 
  void TOFILE(const char* fl_name, const char* msg){}
  // #define TOFILE(fl_name, msg)  
  void CLEARFILE(const char* fl_name){}
  // #define CLEARFILE(fl_name)
#endif


// void my_imshow(const char* win_name, Mat img, bool reset=false);

#endif
