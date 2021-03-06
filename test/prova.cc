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

#define M 2.0
#define startPos 1

const Angle inc(A_2PI.toRad()/M, Angle::RAD);

#define N 6
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
  #if ES==1
  points.add(Point2<TYPE> (0*SCALE,0*SCALE));
  points.add(Point2<TYPE> (-0.1*SCALE,0.3*SCALE));
  points.add(Point2<TYPE> (0.2*SCALE,0.8*SCALE));
  points.add(Point2<TYPE> (1*SCALE,1*SCALE));
  #define _KMAX 3/SCALE

  #elif ES==2 
  points.add(Point2<TYPE> (0*SCALE,0*SCALE));
  points.add(Point2<TYPE> (-0.1*SCALE,0.3*SCALE));
  points.add(Point2<TYPE> (0.2*SCALE,0.8*SCALE));
  points.add(Point2<TYPE> (1*SCALE,1*SCALE));
  points.add(Point2<TYPE> (0.5*SCALE,0.5*SCALE));
  points.add(Point2<TYPE> (0.5*SCALE,0*SCALE));
  #define _KMAX 3/SCALE

  #elif ES==3
  points.add(Point2<TYPE>(0.5*SCALE, 1.2*SCALE));
  points.add(Point2<TYPE>(0.0*SCALE, 0.8*SCALE));
  points.add(Point2<TYPE>(0.0*SCALE, 0.4*SCALE));
  points.add(Point2<TYPE>(0.1*SCALE, 0.0*SCALE));
  points.add(Point2<TYPE>(0.4*SCALE, 0.2*SCALE));

  points.add(Point2<TYPE>(0.5*SCALE, 0.5*SCALE));
  points.add(Point2<TYPE>(0.6*SCALE, 1.0*SCALE));
  points.add(Point2<TYPE>(1.0*SCALE, 0.8*SCALE));
  points.add(Point2<TYPE>(1.0*SCALE, 0.0*SCALE));
  points.add(Point2<TYPE>(1.4*SCALE, 0.2*SCALE));
  
  points.add(Point2<TYPE>(1.2*SCALE, 1.0*SCALE));
  points.add(Point2<TYPE>(1.5*SCALE, 1.2*SCALE));
  points.add(Point2<TYPE>(2.0*SCALE, 1.5*SCALE));
  points.add(Point2<TYPE>(1.5*SCALE, 0.8*SCALE));
  points.add(Point2<TYPE>(1.5*SCALE, 0.0*SCALE));

  points.add(Point2<TYPE>(1.7*SCALE, 0.6*SCALE));
  points.add(Point2<TYPE>(1.9*SCALE, 1.0*SCALE));
  points.add(Point2<TYPE>(2.0*SCALE, 0.5*SCALE));
  points.add(Point2<TYPE>(1.9*SCALE, 0.0*SCALE));
  points.add(Point2<TYPE>(2.5*SCALE, 0.6*SCALE));
  #define _KMAX 5/SCALE

  #endif


  int size=points.size()-1;
  
  #ifdef DEBUG
    cout << "Considered points: " << endl;
    cout << points << endl;
    cout << endl;
  #endif
  
  Tuple<Tuple<Angle> >t;
  Tuple<Angle> z;
  for (int i=startPos; i<size; i++){
    Angle toNext=points.get(i).th(points.get(i+1));
    z.add(toNext-Angle(A_2PI.toRad()/2, Angle::RAD));
  }
  #if ES==1 || ES==2
  z.add(Angle(-(M_PI/6.0), Angle::RAD));
  z.ahead(Angle(-(M_PI/3.0), Angle::RAD));
  #elif ES==3
  z.add(Angle(0, Angle::RAD));
  z.ahead(Angle((5.0*M_PI/6.0), Angle::RAD));
  #endif 

  #ifdef DEBUG
    cout << "Starting angles: " << endl;
    for (auto el : z){
      cout << el.to_string(Angle::DEG).str() << "  ";
    } cout << endl << endl;
  #endif

  disp(t, z, M, inc, startPos, z.size()-2);
  
  COUT(t.size())
  // disp(z, size-1, M, inc, startPos);

  // #ifdef DEBUG
  //   cout << "Considered angles: " << endl;
  //   for (auto tupla : t){
  //     cout << "<";
  //     for (int i=0; i<tupla.size(); i++){
  //       cout << tupla.get(i).to_string(Angle::DEG).str() << (i==tupla.size()-1 ? "" : ", ");
  //     }
  //     cout << ">" << endl;
  //   }
  //   cout << endl;
  // #endif 

  #define DIMX 200
  #define DIMY 200
  #define INC 5
  #define SHIFT 50
  Mat image(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));

  for (auto point : points){
      rectangle(image, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
  }

  my_imshow("dubin", image, true);
  mywaitkey();

  //Compute Dubins
  Tuple<Tuple<Dubins<TYPE> > > allDubins;
  Tuple<Dubins<TYPE> > best;
  double best_l=DInf;
  int count=0;
  double elapsed=0;
  double total=0.0;
  for (auto angleT : t){
    auto start=Clock::now();
    // Mat image(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
    // for (auto point : points){
    //     rectangle(image, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
    // }
    
    Tuple<Dubins<TYPE> > app;
    double l=0.0;
    count += 6;
    for (int i=0; i<angleT.size()-1; i++){
      Dubins<TYPE> d=Dubins<TYPE>(points.get(i), points.get(i+1), angleT.get(i), angleT.get(i+1), _KMAX);
      if (d.getId()<0){
        app=Tuple<Dubins<TYPE> > ();
        break;
      }
      app.add(d);
      l+=d.length();
      #ifdef DEBUG
        // d.draw(1500, 1000, 1, Scalar(255, 0, 0), image);
      #endif      
    }
    auto stop=Clock::now();
    elapsed+=CHRONO::getElapsed(start, stop, CHRONO::MUSEC);
    if (elapsed/1000000.0>5.0){
      total+=elapsed;
      elapsed=0.0;
      COUT(count)
    }
    
    if (best_l>l) {
      best=app; 
      best_l=l;
      #ifdef DEBUG 
        // my_imshow("dubin", image, true);
        // mywaitkey();
      #endif
    }

    allDubins.add(app);
  }
  COUT(total/1000)
  COUT(t.size())
  cout << "All: " << count << endl;
  cout << "Scale: " << elapsedScale << " mus" << endl;
  cout << "Primitives: " << elapsedPrimitives << " mus" << endl;
  cout << "Effective: " << countTries << endl;
  cout << "Best: " << elapsedBest << " mus" << endl;
  cout << "Arcs: " << elapsedArcs << " mus" << endl;
  cout << "Check: " << elapsedCheck << " mus" << endl;
  COUT(elapsedVar)
  COUT(elapsedCirc)
  COUT(elapsedSet)
  COUT(elapsedVar+elapsedCirc+elapsedSet)
  COUT(elapsedTuple)
  COUT(elapsedTupleSet)
  COUT(elapsedLSL)
  COUT(elapsedRSR)
  COUT(elapsedLSR)
  COUT(elapsedRSL)
  COUT(elapsedRLR)
  COUT(elapsedLRL)
  COUT(elapsedLSL+elapsedRSR+elapsedLSR+elapsedRSL+elapsedRLR+elapsedLRL)

  #ifdef DEBUG
    // cout << "Dubins: " << endl;
    // for (auto dub : allDubins){
    //   cout << dub << endl << endl;
    // }
    // cout << endl << endl << endl << endl;

    cout << "Best length: " << best_l << endl;
    for (auto dub : best){
      cout << dub << endl;
    }
  #endif

  Mat best_img(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
  for (auto point : points){
    rectangle(best_img, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
  }
  for (auto dub : best){
    dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img, SHIFT);
  }
  #ifdef DEBUG 
    my_imshow("best", best_img, true);
    mywaitkey();
  #endif

  return 0;
}

