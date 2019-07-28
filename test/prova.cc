#include <iostream>
#include <maths.hh>
#include <dubins.hh>
#include <cmath>

#define DEBUG

using namespace std;

#define M 2.0
extern Tuple<Tuple<Angle> > t;
extern const Angle A_PI;
extern const Angle A_2PI;
extern const Angle A_PI2;
extern const Angle A_RAD_NULL;

const Angle inc(A_PI.toRad()/(2*M), Angle::RAD);

int main(){  
  // cout << std::is_same<Point2<int>, float>::value << endl;
  Tuple<Point2<int> > points;
  points.add(Point2<int> (1,1));
  points.add(Point2<int> (3,3));
  points.add(Point2<int> (5,5));
  points.add(Point2<int> (4,1));
  points.add(Point2<int> (6,5));
  points.add(Point2<int> (8,2));
  points.add(Point2<int> (9,5));
  
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
  z.add(A_RAD_NULL);
  
  #ifdef DEBUG
    cout << "Starting angles: " << endl;
    for (auto el : z){
      cout << el.to_string(Angle::DEG).str() << "  ";
    } cout << endl << endl;
  #endif

  disp(z, size-1, M, inc);
  
  #ifdef DEBUG
    cout << "Considered angles: " << endl;
    for (auto tupla : t){
      cout << "<";
      for (int i=0; i<tupla.size(); i++){
        cout << tupla.get(i).to_string(Angle::DEG).str() << (i==tupla.size()-1 ? "" : ", ");
      }
      cout << ">" << endl;
    }
    cout << "expected: " << pow(M+1, size) << ",  got: " << t.size() << endl;
    cout << endl;
  #endif 

  //Compute Dubins
  Tuple<Dubins<int> > allDubins;
  for (int i=0; i<z.size()-1; i++){
    Dubins<int> best;
    Dubins<int> d=Dubins<int>(points.get(i), points.get(i+1), z.get(i), z.get(i+1));
    if (i==0) best=d;
    if (best.length()>d.length()) best=d;

    allDubins.add(d);
  }

  return 0;
}