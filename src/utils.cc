#include"utils.hh"

void my_imshow( const char* win_name, 
                cv::Mat img, 
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
    #if CV_MAJOR_VERSION<4
      namedWindow(win_name, CV_WINDOW_NORMAL);
    #else 
      namedWindow(win_name, WINDOW_NORMAL);
    #endif

    imshow(win_name, img);
    cvResizeWindow(win_name, SIZE, SIZE);
    moveWindow(win_name, W_now, H_now);

    W_now += SIZE + W_OFFSET;
    if(W_now >= LIMIT){
        W_now = W_0;
        H_now += SIZE + H_OFFSET;
    }
}

void mywaitkey() {
    while((char)waitKey(1)!='q'){}
}

void mywaitkey(string windowName) {
  while((char)waitKey(1)!='q'){}
  cout << "Destroying window " << windowName << endl;
  destroyWindow(windowName);
}

void mywaitkey(Tuple<string> windowNames) {
  while((char)waitKey(1)!='q'){}
  for (auto name : windowNames) {
    destroyWindow(name);
  }
}

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








namespace timeutils {

  int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
  {
    return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
            ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
  }

  double getTimeS()
  {
    static struct timespec t0;
    static int init = 0;
    if (!init)
    {
      clock_gettime(CLOCK_REALTIME, &t0);
      init = 1;
    }
    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    int64_t timediff = timespecDiff(&spec, &t0);
    return timediff/1e9;
//    return ((spec.tv_sec * 1000000000) + spec.tv_nsec)/1e9;
  }

}

#endif
