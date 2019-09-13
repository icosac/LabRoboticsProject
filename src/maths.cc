#include "maths.hh"

// https://stackoverflow.com/questions/9171494/linker-error-when-using-a-template-class

void invertAngle (Angle & a){
  cout << "Inversion of the angle:\n";
  cout << a << endl;
  Point2<double> p0(0.0, 0.0), p1(0.0, 0.0);
  p1.offset(10.0, a);
  
  p0.invert();
  p1.invert();

  a = p0.th(p1);

  cout << a << endl;
}



