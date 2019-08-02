#include <iostream>
#include <maths.hh>
#include <dubins.hh>
// #include <utils.hh>
#include <cmath>

#ifndef DEBUG
#define DEBUG
#endif

using namespace std;

typedef double TYPE;

#define SCALE 100.0

#define ES 3

double elapsedScale=0.0;
double elapsedPrimitives=0.0;
double elapsedBest=0.0;
double elapsedArcs=0.0;
double elapsedCheck=0.0;
double elapsedVar=0.0;
double elapsedCirc=0.0;
double elapsedSet=0.0;
double elapsedTuple=0.0;
double elapsedTupleSet=0.0;
double elapsedLSL=0.0;
double elapsedRSR=0.0;
double elapsedLSR=0.0;
double elapsedRSL=0.0;
double elapsedRLR=0.0;
double elapsedLRL=0.0;
unsigned long countTries=0;

int main(){ 
  Tuple<Point2<TYPE> > points;
  Configuration2<TYPE> start;
  Configuration2<TYPE> stop;

  #if ES==1
  start=Configuration2<TYPE>(0*SCALE,0*SCALE, Angle(-M_PI/3.0, Angle::RAD));
  points.add(Configuration2<TYPE>(-0.1*SCALE,0.3*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.2*SCALE,0.8*SCALE, Angle()));
  stop=Configuration2<TYPE>(1.0*SCALE,1.0*SCALE, Angle(-M_PI/6.0, Angle::RAD));
  #define _KMAX 3/SCALE  

  #elif ES==2 
  start=Configuration2<TYPE>(0*SCALE,0*SCALE, Angle(-M_PI/3.0, Angle::RAD));
  points.add(Configuration2<TYPE>(-0.1*SCALE,0.3*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.2*SCALE,0.8*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.0*SCALE,1.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.5*SCALE,0.5*SCALE, Angle()));
  stop=Configuration2<TYPE>(0.5*SCALE,0.0*SCALE, Angle(-M_PI/6.0, Angle::RAD));
  #define _KMAX 3/SCALE

  #elif ES==3
  start=Configuration2<TYPE>(0.5*SCALE, 1.2*SCALE, Angle(5.0*M_PI/6.0, Angle::RAD));
  points.add(Configuration2<TYPE>(0.0*SCALE, 0.8*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.0*SCALE, 0.4*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.1*SCALE, 0.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.4*SCALE, 0.2*SCALE, Angle()));

  points.add(Configuration2<TYPE>(0.5*SCALE, 0.5*SCALE, Angle()));
  points.add(Configuration2<TYPE>(0.6*SCALE, 1.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.0*SCALE, 0.8*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.0*SCALE, 0.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.4*SCALE, 0.2*SCALE, Angle()));
  
  points.add(Configuration2<TYPE>(1.2*SCALE, 1.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.5*SCALE, 1.2*SCALE, Angle()));
  points.add(Configuration2<TYPE>(2.0*SCALE, 1.5*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.5*SCALE, 0.8*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.5*SCALE, 0.0*SCALE, Angle()));

  points.add(Configuration2<TYPE>(1.7*SCALE, 0.6*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.9*SCALE, 1.0*SCALE, Angle()));
  points.add(Configuration2<TYPE>(2.0*SCALE, 0.5*SCALE, Angle()));
  points.add(Configuration2<TYPE>(1.9*SCALE, 0.0*SCALE, Angle()));
  stop=Configuration2<TYPE>(2.5*SCALE, 0.6*SCALE, Angle());
  #define _KMAX 5/SCALE

  #endif

  DubinsSet<TYPE> ds(start, stop, points, _KMAX);

  return 0;
}

