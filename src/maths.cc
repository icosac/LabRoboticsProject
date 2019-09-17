#include "maths.hh"

// https://stackoverflow.com/questions/9171494/linker-error-when-using-a-template-class

/*! \brief Transform the angle given i the new reference system where x and y will be swapped.
    \param[in/out] a The angle that need to be inverted.
*/
void invertAngle (Angle & a){
  Point2<double> p0(0.0, 0.0), p1(0.0, 0.0);
  p1.offset(10.0, a);
  
  p0.invert();
  p1.invert();

  a = p0.th(p1);
}



