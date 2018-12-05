#include"utils.hh"

void my_imshow( const char* win_name, 
                Mat img, 
                bool reset/*=false*/){
    const int SIZE     = 250;
    const int W_0      = 0;
    const int H_0      = 0;
    const int W_OFFSET = 20;
    const int H_OFFSET = 90;
    const int LIMIT    = W_0 + 5*SIZE + 4*W_OFFSET;

    static int W_now = W_0;
    static int H_now = H_0;
    if(reset){
        W_now = W_0;
        H_now = H_0;
    }

    //const string s = win_name;
    namedWindow(win_name, CV_WINDOW_NORMAL);
    cvvResizeWindow(win_name, SIZE, SIZE);
    imshow(win_name, img);
    moveWindow(win_name, W_now, H_now);
    W_now += SIZE + W_OFFSET;
    if(W_now >= LIMIT){
        W_now = W_0;
        H_now += SIZE + H_OFFSET;
    }
}

#ifdef DEBUG

#endif

#if defined PRINT_TO_FILE && defined DEBUG

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
  
  void TOFILE(const char* fl_name, const char* msg){}
  // #define TOFILE(fl_name, msg)  
  void CLEARFILE(const char* fl_name){}
  // #define CLEARFILE(fl_name)

#endif