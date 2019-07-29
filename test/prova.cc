#include <iostream>
#include <maths.hh>
#include <dubins.hh>
#include <utils.hh>
#include <cmath>

#define DEBUG

using namespace std;

typedef double TYPE;

#define M 4.0
#define startPos 1
extern Tuple<Tuple<Angle> > t;

extern Angle A_PI;
extern Angle A_2PI;
extern Angle A_PI2;
extern Angle A_90;
extern Angle A_RAD_NULL;

const Angle inc(A_PI.toRad()/(2*M), Angle::RAD);

int main(){  
  Tuple<Point2<TYPE> > points;
  // points.add(Point2<TYPE> (1*100,1*100));
  points.add(Point2<TYPE> (3*100,3*100));
  // points.add(Point2<TYPE> (5*100,5*100));
  points.add(Point2<TYPE> (4*100,1*100));
  points.add(Point2<TYPE> (4*100,2*100));
  points.add(Point2<TYPE> (6*100,5*100));
  points.add(Point2<TYPE> (8*100,2*100));
  points.add(Point2<TYPE> (9*100,5*100));
  
  int size=points.size()-1;
  
  #ifdef DEBUG
    cout << "Considered points: " << endl;
    cout << points << endl;
    cout << endl;
  #endif
  
  Tuple<Angle> z;
  for (int i=0; i<size; i++){
    Angle toNext=points.get(i).th(points.get(i+1));
    z.add(toNext-Angle(A_PI2.toRad()/2, Angle::RAD));
  }
  z.add(Angle(270, Angle::DEG));

  points.ahead(Point2<TYPE> (1*100,1*100));
  z.ahead(Angle(315, Angle::DEG));

  #ifdef DEBUG
    cout << "Starting angles: " << endl;
    for (auto el : z){
      cout << el.to_string(Angle::DEG).str() << "  ";
    } cout << endl << endl;
  #endif

  disp(z, size-1, M, inc, startPos);

  #ifdef DEBUG
    cout << "Considered angles: " << endl;
    for (auto tupla : t){
      cout << "<";
      for (int i=0; i<tupla.size(); i++){
        cout << tupla.get(i).to_string(Angle::DEG).str() << (i==tupla.size()-1 ? "" : ", ");
      }
      cout << ">" << endl;
    }
    cout << "expected: " << pow(M+1, size-startPos) << ",  got: " << t.size() << endl;
    cout << endl;
  #endif 

  #define DIMX 1000
  #define DIMY 650
  #define INC 35
  Mat image(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));

  for (auto point : points){
      rectangle(image, Point(point.x(), point.y()), Point(point.x()+INC, point.y()+INC), Scalar(0,0,0) , -1);
  }

  my_imshow("dubin", image, true);
  mywaitkey();

  //Compute Dubins
  Tuple<Tuple<Dubins<TYPE> > > allDubins;
  Tuple<Dubins<TYPE> > best;
  double best_l=DInf;
  for (auto angleT : t){

    Mat image(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
    for (auto point : points){
        rectangle(image, Point(point.x()-INC/2, point.y()-INC/2), Point(point.x()+INC/2, point.y()+INC/2), Scalar(0,0,0) , -1);
    }
    
    Tuple<Dubins<TYPE> > app;
    double l=0.0;
    for (int i=0; i<angleT.size()-1; i++){
      Dubins<TYPE> d=Dubins<TYPE>(points.get(i), points.get(i+1), angleT.get(i), angleT.get(i+1), 0.01);
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
    rectangle(best_img, Point(point.x()-INC/2, point.y()-INC/2), Point(point.x()+INC/2, point.y()+INC/2), Scalar(0,0,0) , -1);
  }
  for (auto dub : best){
    dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img);
  }
  #ifdef DEBUG 
    my_imshow("best", best_img, true);
    mywaitkey();
  #endif

  return 0;
}