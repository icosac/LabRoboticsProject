#ifndef UTILS_HH
#define UTILS_HH

#ifdef TESS
#include <tesseract/baseapi.h> // Tesseract headers
#include <leptonica/allheaders.h>
#endif

#include <sstream>
#include <iostream>
#include <exception>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

typedef chrono::high_resolution_clock Clock;
namespace CHRONO {
  enum TIME_TYPE {SEC, MSEC, MUSEC, NSEC};

  inline string getType(TIME_TYPE type, string ret=""){
    switch(type){
      case SEC: return ret+"s";
      case MSEC: return ret+"ms";
      case MUSEC: return ret+"mus";
      case NSEC: return ret+"ns";
    }
    return "";
  }

  inline double getElapsed(Clock::time_point start, 
                    Clock::time_point stop,
                    TIME_TYPE type=MUSEC){
    switch(type){
      case SEC:{
        return chrono::duration_cast<chrono::seconds>(stop - start).count();
      }
      case MSEC:{
        return chrono::duration_cast<chrono::milliseconds>(stop - start).count();
      }
      case MUSEC:{
        return chrono::duration_cast<chrono::nanoseconds>(stop - start).count()/1000.0;
      }
      case NSEC:{
        return chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
      }
    }
    return 0.0;
  }

  inline string getElapsed(Clock::time_point start,
                    Clock::time_point stop,
                    string ret,
                    TIME_TYPE type=MUSEC){
    return ret+to_string(getElapsed(start, stop, type))+getType(type, " ");
  }
}

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

/*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed.
 *
 */
void mywaitkey(const char c='q');

/*!\brief Function to use after my_imshow() for keeping the image opened until a key is pressed. When a key is pressed a specific window is closed.
 *
 * @param windowName The window to close after pressing a key.
 */
void mywaitkey(string windowName);

enum EXCEPTION_TYPE {GENERAL, EXISTS, SIZE}; ///< The type of the exceptions. GENERAL allows to pass a string with a general exception. EXISTS should be used in case an element already exists. SIZE should be used when there is a problem with vector's size.

/*!
 * This class allows to throw personalized exceptions. 
 * \tparam T The type of a variable.
 */
template<class T>
class MyException : public exception {
private:
  /**
   * \brief      A function to convert a value of type T in a string
   * \param[in]  value  The value to convert.
   * \tparam     T1 The type of the value to convert.
   * \return     A string containing the value.
   */
  template<class T1>
  stringstream exceptString(T1 value) const {
    stringstream out;
    out << value;
    return out;
  }
  
public:
  EXCEPTION_TYPE type;
  T a;
  int b;
  string s;
  /**
   * \brief      Plain constructor for the object.
   *
   * \param[in]  _type  The type of the exception
   * \param[in]  _a     Variable meaning.
   * \param[in]  _b     Variable meaning.
   * \param[in]  _s     Variable meaning.
   */
  MyException(EXCEPTION_TYPE _type, T _a, int _b, string _s = "???") : type(_type), a(_a), b(_b), s(_s){}
  
  /**
   * \brief      Function to call to get the exception meaning. 
   *
   * \return     A string containing why the exception was thrown.
   */
  const char * what() const throw (){
    string ret;
    switch(type){
      case GENERAL: {
        ret=NAME(type)+string("_Exception: ")+exceptString(a).str()+" at line: "+exceptString(b).str()+" in file: "+exceptString(s).str();
        break;
      }
      case EXISTS:{
        ret=NAME(type)+string("_Exception: element already exists: ")+exceptString(a).str()+" at pos: "+exceptString(b).str();
        break;
      }
      case SIZE:{
        ret=NAME(tyoe)+string("_Exception: sizes are different: ")+exceptString(a).str()+"!="+exceptString(b).str();
      }
    }
    cout << ret.c_str() << endl;
    return "";
  }
};

//DEFAULT FOR EXAM
#include <time.h>
#include <cstdint>

namespace timeutils {

  int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p);
  double getTimeS();

}

#endif
