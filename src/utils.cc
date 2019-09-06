#include<utils.hh>

/*! \brief Function to show images in an order grill.
 * @param win_name The name of the window to use.
 * @param img The Mat containing the image.
 * @param reset If true the image is going to be placed in 0,0 i.e. the top left corner of the screen.
 */
// void my_imshow( const char* win_name,
//                 cv::Mat img, 
//                 bool reset/*=false*/){
//     const int SIZE     = 250;
//     const int W_0      = 0;
//     const int H_0      = 0;
//     const int W_OFFSET = 20;
//     const int H_OFFSET = 90;
//     const int LIMIT    = W_0 + 5*SIZE + 4*W_OFFSET;

//     static int W_now = W_0;
//     static int H_now = H_0;
//     if(reset){
//         W_now = W_0;
//         H_now = H_0;
//     }

//     //const string s = win_name;
//     #if CV_MAJOR_VERSION<4
//       namedWindow(win_name, CV_WINDOW_AUTOSIZE);
//     #else 
//       namedWindow(win_name, WINDOW_NORMAL);
//     #endif

//     imshow(win_name, img);
//     cvResizeWindow(win_name, SIZE, SIZE);
//     moveWindow(win_name, W_now, H_now);

//     W_now += SIZE + W_OFFSET;
//     if(W_now >= LIMIT){
//         W_now = W_0;
//         H_now += SIZE + H_OFFSET;
//     }
// }

// /*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed.
//  *
//  */
 
// void mywaitkey() {
//     while((char)waitKey(1)!='q'){}
// }

// /*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed. When a key is pressed a specific window is closed.
//  *
//  * @param windowName The window to close after pressing a key.
// */ 
// void mywaitkey(string windowName) {
//   while((char)waitKey(1)!='q'){}
//   cout << "Destroying window " << windowName << endl;
//   destroyWindow(windowName);
// }


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