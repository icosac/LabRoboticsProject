#pragma once 
#ifndef DUBINS_HH
#define DUBINS_HH

// #include <utils.hh>
#include <maths.hh>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#if defined DEBUG && defined REALLY_DEBUG
#include <cstdio> // For sprintf
#endif

extern double elapsedScale;
extern double elapsedPrimitives;
extern double elapsedBest;
extern double elapsedArcs;
extern double elapsedCheck;
extern unsigned long countTries;
extern double elapsedVar;
extern double elapsedCirc;
extern double elapsedSet;
extern double elapsedLSL;
extern double elapsedRSR;
extern double elapsedLSR;
extern double elapsedRSL;
extern double elapsedRLR;
extern double elapsedLRL;

#ifdef DEBUG
unsigned int INC=5;
unsigned int SHIFT=100;
unsigned int DIMX=200+SHIFT;
unsigned int DIMY=500+SHIFT;
#endif

//TODO find which function is faster
#define MORE_FUNCTIONS
#define PIECE_LENGTH 2 //mm
#define PREC 100000

using namespace std;

template <class T>
class Curve {
protected:
  Configuration2<T> P0;
  Configuration2<T> P1;

public:
  Curve () : P0(), P1() {}
  Curve (const Configuration2<T> _P0,
         const Configuration2<T> _P1) :
  P0(_P0), P1(_P1) {}

  Curve (const Point2<T> _P0,
         const Point2<T> _P1,
         const Angle _th0,
         const Angle _th1) :
  P0(_P0, _th0), P1(_P1, _th1) {}

  Curve (const T x0, const T y0,
         const Angle _th0,
         const T x1, const T y1,
         const Angle _th1) :
  P0(x0, y0, _th0), P1(x1, y1, _th1) {}

  Configuration2<T> begin()  const { return P0; }
  Configuration2<T> end()    const { return P1; }
  
#ifdef MORE_FUNCTIONS
  void begin(Configuration2<T> _P0){
    P0=_P0;
  }
  void end (Configuration2<T> _P1){
    P1=_P1;
  }
#endif
  
  friend ostream& operator<<(ostream &out, const Curve& data) {
    out << data.to_string().str();
    return out;
  }
  
  stringstream to_string() const{
    stringstream out;
    out << "begin: " << begin();
    out << ", end: " << end();
    return out;
  }
};

static double sinc(double t) {
  if (std::abs(t)<0.002)
    return 1 - pow2(t)/6 * (1 - pow2(t)/20);
  else
    return sin(t)/t;
}

//Computes an arrival point from an initial configuration through an arc of length _L and curvature _K.
Configuration2<double> circline(double _L,
                                Configuration2<double> _P0,
                                double _K)
{
  double app=_K*_L/2.0;
  double sincc=_L*sinc(app);
  double phi=_P0.angle().toRad();
  
  double x=_P0.x() + sincc * cos(phi+app);
  double y=_P0.y() + sincc * sin(phi+app);
  Angle th=Angle(_K*_L+phi, Angle::RAD);

  return Configuration2<double>(x, y, th.get());
}

template <class T1=double, class T2=double>
class DubinsArc : public Curve<T2>
{
private:
  T1 L, K;

  using Curve<T2>::Curve;
public:

  DubinsArc () : L(0), K(0), Curve<T2>() {}

#ifdef MORE_FUNCTIONS
  DubinsArc( const Configuration2<T2> _P0,
                  const T1 _k,
                  const T1 _l) : Curve<T2>() {
    auto start=Clock::now();
    K=_k;
    L=_l;
    auto stop=Clock::now();
    elapsedVar+=CHRONO::getElapsed(start, stop);

    start=Clock::now();
    Configuration2<T2> _P1 = circline(L, _P0, K);
    stop=Clock::now();
    elapsedCirc+=CHRONO::getElapsed(start, stop);

    start=Clock::now();
    Curve<T2>::begin(_P0); Curve<T2>::end(_P1);
    stop=Clock::now();
    elapsedSet+=CHRONO::getElapsed(start, stop);
  }
#else
  DubinsArc(const Configuration2<T2> _P0,
            const Configuration2<T2> _P1,
            const T1 _k,
            const T1 _l) : Curve<T2>(_P0, _P1) {
    K=_k;
    L=_l;
    cout << "_P0: " << _P0 << endl;
    cout << "begin: " << Curve<T2>::begin() << endl;
    cout << "_P1: " << _P1 << endl;
    cout << "end: " << Curve<T2>::end() << endl;
  }
#endif

  T1 getK   () const { return K; }
  T1 length () const { return L; }

  //Splits arc in pieces of _L length
  //TODO Add last point of curve
  Tuple<Point2<T2> > splitIt (double _L=PIECE_LENGTH){
    Tuple<Point2<T2> > ret;
    Configuration2<T2> _old=Curve<T2>::begin();
    double sum=0;

    ret.add(_old); 

    while( length()>sum+_L ){
      Configuration2<T2> _new=circline(_L, _old, getK());
      ret.add(_new);
      _old=_new; //Maybeeeee using pointers can improve performance?
      sum+=_L;
    }

    // INFOV(ret)

    return ret;
  }

  stringstream to_string() const {
    stringstream out;
    out << "begin: " << Curve<T2>::begin();
    out << ", end: " << Curve<T2>::end();
    out << ", K: " << getK();
    out << ", l: " << length();
    return out;
  }
  
  friend ostream& operator<<(ostream &out, const DubinsArc& data) {
    out << data.to_string().str();
    return out;
  }

  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image, double SHIFT){
    // Mat imageMap(dimX, dimY, CV_8UC3, Scalar(255,255,255));
    for (auto point : this->splitIt(1)){
      if (point.x()>dimX || point.y()>dimY){
        double x=point.x()>dimX ? point.x() : dimX;
        double y=point.y()>dimY ? point.y() : dimY;
        Mat newMat(x, y, CV_8UC3, Scalar(255, 255, 255));
        for (double _x=0; _x<dimX; _x++){
          for (double _y=0; _y<dimY; _y++){
            rectangle(newMat, Point(_x+SHIFT, _y+SHIFT),Point(_x+inc+SHIFT, _y+inc+SHIFT), scl, -1);
          }
        }
        image=newMat;
      }
      rectangle(image, Point(point.x()+SHIFT, point.y()+SHIFT), Point(point.x()+inc+SHIFT, point.y()+inc+SHIFT), scl, -1);
    }
  }

};


#define KMAX 0.5
template<class T>
class Dubins : protected Curve<T>
{
private:
  double Kmax, L;
  int pid;
  DubinsArc<T> A1, A2, A3;

  using Curve<T>::Curve;
  // using DubinsArc<T>::DubinsArc;
public:
  Dubins () : Kmax(KMAX), Curve<T>() {
    A1=DubinsArc<T>();
    A2=DubinsArc<T>();
    A3=DubinsArc<T>();
  }

  Dubins (const Configuration2<T> _P0,
          const Configuration2<T> _P1,
          const double _K=KMAX) :
  Curve<T>(_P0, _P1), Kmax(_K) {
    pid=shortest_path();
    if (pid<0){
      A1=DubinsArc<T>();
      A2=DubinsArc<T>();
      A3=DubinsArc<T>();
    }
  }

  Dubins (const Point2<T> _P0,
          const Point2<T> _P1,
          const Angle _th0,
          const Angle _th1,
          const double _K=KMAX) :
  Curve<T>(_P0, _P1, _th0, _th1), Kmax(_K) {
    pid=shortest_path();
    if (pid<0){
      A1=DubinsArc<T>();
      A2=DubinsArc<T>();
      A3=DubinsArc<T>();
    }
  }

  Dubins (const T x0, const T y0,
          const Angle _th0,
          const T x1, const T y1,
          const Angle _th1,
          const double _K=KMAX) :
  Curve<T>(x0, y0, _th0, x1, y1, _th1), Kmax(_K) {
    pid=shortest_path();
    if (pid<0){
      A1=DubinsArc<T>();
      A2=DubinsArc<T>();
      A3=DubinsArc<T>();
    }
  }

  double getKMax  () const { return Kmax; }
  double length   () const { return L; }
  double getId    ()  { return pid; }

  DubinsArc<T> getA1() const { return A1; }
  DubinsArc<T> getA2() const { return A2; }
  DubinsArc<T> getA3() const { return A3; }

#ifndef OPENCL_COMPILE
  double* LSL (double th0, double th1, double _kmax)
  {
    auto start=Clock::now();
    double C=cos(th1)-cos(th0);
    double S=2*_kmax+sin(th0)-sin(th1);
    double tan2=atan2(C, S);
    
    double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1));
    
    if (temp1<0){
      // return Tuple<double> (0);
      return nullptr;
    }
    
    double invK=1/_kmax;
    double sc_s1=Angle(tan2-th0, Angle::RAD).get()*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=Angle(th1-tan2, Angle::RAD).get()*invK;
    
    auto stop=Clock::now();
    elapsedLSL+=CHRONO::getElapsed(start, stop);

    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1.get(), sc_s2, sc_s3.get());
  }

  double* RSR (double th0, double th1, double _kmax)
  {
    auto start=Clock::now();
    double C=cos(th0)-cos(th1);
    double S=2*_kmax-sin(th0)+sin(th1);
    
    double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1));
    
    if (temp1<0){
      // return Tuple<double> (0);
      return nullptr;
    }
    
    double invK=1/_kmax;
    double sc_s1=Angle(th0-atan2(C,S), Angle::RAD).get()*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=Angle(atan2(C,S)-th1, Angle::RAD).get()*invK;
    
    auto stop=Clock::now();
    elapsedRSR+=CHRONO::getElapsed(start, stop);

    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    
    // return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  double* LSR (double th0, double th1, double _kmax)
  {    
    auto start=Clock::now();
    double C = cos(th0)+cos(th1);
    double S=2*_kmax+sin(th0)+sin(th1);
    
    double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)+sin(th1));
    if (temp1<0){
      // return Tuple<double> (0);
      return nullptr;
    }
    
    double invK=1/_kmax;
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0, Angle::RAD).get()*invK;
    double sc_s3= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1, Angle::RAD).get()*invK;

    auto stop=Clock::now();
    elapsedLSR+=CHRONO::getElapsed(start, stop);

    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  double* RSL (double th0, double th1, double _kmax)
  {
    auto start=Clock::now();
    double C = cos(th0)+cos(th1);
    double S=2*_kmax-sin(th0)-sin(th1);
    
    double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)+sin(th1));
    if (temp1<0){
      // return Tuple<double> (0);
      return nullptr;
    }
    
    double invK=1/_kmax;
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1= Angle(th0-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
    double sc_s3= Angle(th1-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
    
    auto stop=Clock::now();
    elapsedRSL+=CHRONO::getElapsed(start, stop);
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  double* RLR (double th0, double th1, double _kmax)
  {
    auto start=Clock::now();
    double C=cos(th0)-cos(th1);
    double S=2*_kmax-sin(th0)+sin(th1);
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));
    
    if (fabs(temp1)-Epsi>1.0){
      // return Tuple<double> (0);
      return nullptr;
    }
    
    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(th0-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th0-th1+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    auto stop=Clock::now();
    elapsedRLR+=CHRONO::getElapsed(start, stop);
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  double* LRL (double th0, double th1, double _kmax)
  {
    auto start=Clock::now();
    double C=cos(th1)-cos(th0);
    double S=2*_kmax+sin(th0)-sin(th1);
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

    if (fabs(temp1)-Epsi>1.0){
      // return Tuple<double> (0);
      return nullptr;
    }

    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(atan2(C, S)-th0+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th1-th0+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    auto stop=Clock::now();
    elapsedLRL+=CHRONO::getElapsed(start, stop);
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;

    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }
#endif

  Tuple<double> scaleToStandard ()
  {
    double dx=Curve<T>::end().x() - Curve<T>::begin().x();
    double dy=Curve<T>::end().y() - Curve<T>::begin().y();

    double _phi=atan2(dy, dx);

    Angle phi= Angle(_phi, Angle::RAD);

    double lambda=sqrt(pow2(dx)+pow2(dy))/2; //hypt

    Angle sc_th0 = Curve<T>::begin().angle()-phi;
    Angle sc_th1 = Curve<T>::end().angle()-phi;

    double sc_Kmax = Kmax*lambda;

    return Tuple<double> (4, sc_th0.toRad(), sc_th1.toRad(), sc_Kmax, lambda);
  }

  Tuple<double> scaleFromStandard(double lambda,
                                  double sc_s1,
                                  double sc_s2,
                                  double sc_s3){
    return Tuple<double> (3,  (sc_s1 * lambda),
                              (sc_s2 * lambda),
                              (sc_s3 * lambda));
  }

  int shortest_path()
  {
    int pidx=-1; //Return value
    auto start=Clock::now();
    Tuple<double> scaled = scaleToStandard();
    auto stop=Clock::now();
    // cout << CHRONO::getElapsed(start, stop, "scaleToStandard: ") << endl;
    elapsedScale+=CHRONO::getElapsed(start, stop);

    Angle  sc_th0     =  Angle(scaled.get(0), Angle::RAD);
    Angle  sc_th1     =  Angle(scaled.get(1), Angle::RAD); 
    double sc_Kmax    =  scaled.get(2);
    double sc_lambda  =  scaled.get(3);

    double Length = DInf;
    double sc_s1  = 0.0;
    double sc_s2  = 0.0;
    double sc_s3  = 0.0;

    start=Clock::now();
    Tuple<double* > res;
    res.add(LSL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RSR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(LSR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RSL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RLR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(LRL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    stop=Clock::now();
    // cout << CHRONO::getElapsed(start, stop, "Compute primitives: ") << endl;
    elapsedPrimitives+=CHRONO::getElapsed(start, stop);

    for (auto t : res){
      if (t!=nullptr)
        printf("MAH %f %f %f %f %f %f\n", sc_th0, sc_th1, sc_Kmax, t[0], t[1], t[2]);
      else 
        printf("MAH %f %f %f nullptr\n", sc_th0, sc_th1, sc_Kmax);
    }

    int i=0; 
    start=Clock::now(); 
    for (auto value : res){
      if (value!=nullptr){
        double appL=value[0]+value[1]+value[2];
        if (appL<Length){
          Length = appL;
          sc_s1=value[0];
          sc_s2=value[1];
          sc_s3=value[2];
          pidx=i;
        }
      }

      // if (value.size()>0){
      //   double appL=value.get(0)+value.get(1)+value.get(2);
      //   if (appL<Length){
      //     Length = appL;
      //     sc_s1=value.get(0);
      //     sc_s2=value.get(1);
      //     sc_s3=value.get(2);
      //     pidx=i;
      //   }
      // }
      i++;
    }

    if (pidx>=0){
      countTries++;
      Tuple<double> sc_std = scaleFromStandard(sc_lambda, sc_s1, sc_s2, sc_s3);
      vector<vector<int> > ksigns ={
        { 1,  0,  1}, // LSL
        {-1,  0, -1}, // RSR
        { 1,  0, -1}, // LSR
        {-1,  0,  1}, // RSL
        {-1,  1, -1}, // RLR
        { 1, -1,  1}  // LRL
      };

      stop=Clock::now();
      // cout << CHRONO::getElapsed(start, stop, "Choose best ") << endl;
      elapsedBest+=CHRONO::getElapsed(start, stop);

      start=Clock::now();
#ifdef MORE_FUNCTIONS
      A1=DubinsArc<T>(Curve<T>::begin(), ksigns[pidx][0]*Kmax, sc_std.get(0));
      A2=DubinsArc<T>(A1.end(), ksigns[pidx][1]*Kmax, sc_std.get(1));
      A3=DubinsArc<T>(A2.end(), ksigns[pidx][2]*Kmax, sc_std.get(2));
#else
      double L = sc_std.get(0);
      double K = ksigns[pidx][0]*Kmax;
      Configuration2<double> _P1 = circline(L, Curve<T>::begin(), K);
      COUT(_P1)  
      A1=DubinsArc<T>(Curve<T>::begin(), _P1, K, L);
      
      L = sc_std.get(1); K = ksigns[pidx][1]*Kmax;
      _P1 = circline(L, A1.end(), K);
      A2=DubinsArc<T>(A1.end(), _P1, K, L);
      
      L = sc_std.get(2); K = ksigns[pidx][2]*Kmax;
      _P1 = circline(L, A2.end(), K);
      A3=DubinsArc<T>(A2.end(), _P1, K, L);
#endif
      stop=Clock::now();
      // cout << CHRONO::getElapsed(start, stop, "Create arcs ") << endl;
      elapsedArcs+=CHRONO::getElapsed(start, stop);

      start=Clock::now();
      L=A1.length()+A2.length()+A3.length(); //Save total length of Dubins curve

      bool check_ = check(sc_s1, ksigns[pidx][0]*sc_Kmax,
                          sc_s2, ksigns[pidx][1]*sc_Kmax,
                          sc_s3, ksigns[pidx][2]*sc_Kmax,
                          sc_th0, // Curve<T>::begin().angle(),
                          sc_th1  // Curve<T>::end().angle()
                        );
      if (!check_)
        pidx=-2;

      stop=Clock::now();
      elapsedCheck+=CHRONO::getElapsed(start, stop);

    }
    
    
    return pidx;
  }

  bool check (double s1,
              double k0,
              double s2,
              double k1,
              double s3,
              double k2,
              Angle th0,
              Angle th1) const 
  {
    int x0 = -1;
    int y0 = 0;
    int x1 = 1;
    int y1 = 0;

    double eq1  = x0 + s1 * sinc((0.5) * k0 * s1) * (th0+Angle((0.5)*k0*s1, Angle::RAD)).cos() +
                  s2 * sinc((0.5) * k1 * s2) * (th0+Angle(k0*s1, Angle::RAD)+Angle(0.5*k1*s2, Angle::RAD)).cos() +
                  s3 * sinc((0.5) * k2 * s3) * (th0+Angle(k0*s1, Angle::RAD)+Angle(k1*s2, Angle::RAD)+Angle(0.5*k2*s3, Angle::RAD)).cos() -x1;
    double eq2  = y0 + s1 * sinc((0.5) * k0 * s1) * (th0+Angle((0.5)*k0*s1, Angle::RAD)).sin() +
                  s2 * sinc((0.5) * k1 * s2) * (th0+Angle(k0*s1, Angle::RAD)+Angle(0.5*k1*s2, Angle::RAD)).sin() +
                  s3 * sinc((0.5) * k2 * s3) * (th0+Angle(k0*s1, Angle::RAD)+Angle(k1*s2, Angle::RAD)+Angle(0.5*k2*s3, Angle::RAD)).sin() -y1;
    double eq3 =  rangeSymm(k0 * s1 + k1 * s2 + k2 * s3 + th0.get() - th1.get());

    return ((sqrt(pow2(eq1)+pow2(eq2)+pow2(eq3)) < 1.e-6) &&
            (s1>0 || s2>0 || s3>0));
  }

  //Normalize an angular difference \f$(-\pi, \pi]\f$
  static double rangeSymm (double ang){
    double out = ang;
    while (out <= -M_PI){
      out = out + 2 * M_PI;
    }
    while (out > M_PI){
      out = out - 2 * M_PI;
    }
    return out;
  }

  //TODO there are two points that are useless. 
  Tuple<Tuple<Point2<double> > > splitIt (int _arch=0, 
                                          double _L=PIECE_LENGTH){
    Tuple<Tuple<Point2<double> > > v;
    switch(_arch){
      case 1: {
        v.add(A1.splitIt(_L));
        break;
      }
      case 2: {
        v.add(A2.splitIt(_L));
        break;
      }
      case 3: {
        v.add(A3.splitIt(_L));
        break;
      }
      default: {
        v.add(A1.splitIt(_L));
        v.add(A2.splitIt(_L));
        v.add(A3.splitIt(_L));
      }
      return v;
    }
  }

  stringstream to_string() const {
    stringstream out;
    out << "A1: " << getA1() << endl;
    out << "A2: " << getA2() << endl;
    out << "A3: " << getA3();
    return out;
  }

  friend ostream& operator<<(ostream &out, const Dubins& data) {
    out << data.to_string().str();
    return out;
  }

  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image, double SHIFT=0){
    A1.draw(dimX, dimY, inc, scl, image, SHIFT);
    A2.draw(dimX, dimY, inc, scl, image, SHIFT);
    A3.draw(dimX, dimY, inc, scl, image, SHIFT);
  }

};

// #define EXP
#ifdef EXP
//TODO find non recursive approach
/*! \brief Compute the arrangements.
 */
void disp ( Tuple<Tuple<Angle> >& t,
            Tuple<Angle>& z,    ///<Vector to use
            int id,             ///<Position on the vector to change
            int N,              ///<Number of time to "iterate"
            const Angle& inc,   ///<Incrementation
            int startPos=0)        ///<If there are values at the beginning of the tuple not to change.
// const Angle& start) ///<Starting `Angle`
{
  if (id==startPos){
    Angle start=z.get(id);
    for (int i=0; i<N; i++){
      t.addIfNot(z);
      // a+=inc;
      z.set(id, z.get(id)+inc);
      if (i==N-1){
        t.addIfNot(z);
      }
    }
    z.set(id, start);
  }
  else {
    Angle start=z.get(id);
    for (int i=0; i<N; i++){
      disp(t, z, id-1, N, inc, startPos);
      // a+=inc;
      z.set(id, z.get(id)+inc);
      if (i==N-1)
        disp(t, z, id-1, N, inc, startPos);
    }
    z.set(id, start);
  }
}
#else

Tuple<Angle> toBase(Tuple<Angle> z, int n, int base, const Angle& inc, int startPos, int endPos){
  int i=z.size()-1;
  do {
    if (i<startPos || i>endPos){}
    else {
      z.set(i, (z.get(i)+Angle(inc.toRad()*(n%base), Angle::RAD)));
      n=(int)(n/base);
    }
    i--;
  } while(n!=0 && i>-1);

  return z;
}

/*! \brief Compute the arrangements.
 */
void disp(Tuple<Tuple<Angle> >& t,
          Tuple<Angle>& z,    ///<Vector to use
          int N,              ///<Number of time to "iterate"
          const Angle& inc,   ///<Incrementation
          int startPos=0, 
          int endPos=0){
  unsigned long M=z.size()-startPos;
  COUT(z.size());
  if (endPos>startPos){
    M-=(z.size()-endPos-1);
  }
  unsigned long iter_n=pow(N, M);
  COUT(inc)
  COUT(N)
  COUT(M)
  COUT(iter_n)
  COUT(z.size())
  COUT(startPos)
  COUT(endPos)
  for (unsigned long i=0; i<iter_n; i++){
    #ifdef DEBUG
      Tuple<Angle> app=toBase(z, i, N, inc, startPos, endPos);
      t.add(app);
      // COUT(app)
    #else
      t.add(toBase(z, i, N, inc));
    #endif
  }
  cout << "Expected: " << iter_n << " got: " << t.size() << endl;
  // for (auto T : t) {
  //   COUT(T)
  // }

}
#endif

/*!\brief Given a set of point, compute the shortest set of Dubins that allows to go from start to end through all points.
 *
 */
template <class T>
class DubinsSet {
private: 
  Tuple<Dubins<T> > dubinses;
  double Kmax, L;
public:
  DubinsSet(Tuple<Dubins<T> > _dubinses,
            double _kmax=KMAX){
    this->dubinses=_dubinses;
    this->Kmax=_kmax;
    for (Dubins<T> dub : this->dubinses){
      this->L+=dub.length();
    } 
  }

  DubinsSet(Tuple<Configuration2<T> > _confs,
            double _kmax=KMAX){
    for (int i=0; i<_confs.size()-1; i++){
      Dubins<T> dub=Dubins<T>(_confs.get(i), _confs.get(i+1));
      this->dubinses.add(dub);
      this->L+=dub.length();
    }
    this->Kmax=_kmax;
  }

  DubinsSet(Configuration2<T> start, 
            Configuration2<T> end,
            Tuple<Point2<T> > _points,
            double _kmax=KMAX){
    Tuple<Angle> angles;

    angles.add(start.angle());
    for (int i=0; i<_points.size()-1; i++){
      angles.add(_points.get(i).th(_points.get(i+1)));
    }
    angles.add(_points.get(_points.size()-1).th(end.point()));

    angles.add(end.angle());
    _points.ahead(start.point());
    _points.add(end.point());

    Angle area=A_2PI;
    
    int i=0;
    while((int)(area.toRad()*PREC)%PREC>1 && i<1){
      COUT(angles)
      find_best(_points, angles, area, 6.0, _kmax);
      area=area/6.0;
      i++;
    }

    #ifdef DEBUG 
      Mat best_img(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
      for (auto point : _points){
        rectangle(best_img, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
      }
      for (auto dub : this->dubinses){
        dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img, SHIFT);
      }
      my_imshow("best", best_img, true);
      mywaitkey();
      cout << *this << endl;
    #endif
  }

  DubinsSet(Tuple<Point2<T> > _points,
            Angle area,
            int tries,
            double _kmax=KMAX){
    find_best(_points, Tuple<Angle>(), area, tries, _kmax);
  }

  void find_best( Tuple<Point2<T> > _points,
                          Tuple<Angle>& _angles,
                          Angle area=A_2PI,
                          double tries=2.0,
                          double _kmax=KMAX){
 
    #ifdef DEBUG
      cout << "Considered points: " << endl;
      cout << _points << endl;
      cout << endl;
    #endif

    //Compute all initial angles, that is the coeficient for the line that connects two points
    //Even though this is not a precise guess, still is efficient to first consider this angles instead of 0pi.
    
    #ifdef DEBUG
      cout << "Starting angles: " << endl;
      for (auto el : _angles){
        cout << el.to_string(Angle::DEG).str() << "  ";
      } cout << endl << endl;
    #endif
    
    //Compute inc:
    Angle inc=area/tries;
    COUT(inc)
    Tuple<Tuple<Angle> > angles;

    //Create all angles to check
    disp(angles, _angles, tries, inc, 1, _points.size()-2); //startPos=1 and endPos=size()-2 since I have to check for all angles except the first and the last.

    #ifdef DEBUG
      cout << "Considered angles: " << endl;
      // for (auto tupla : angles){
      //   cout << "<";
      //   for (int i=0; i<tupla.size(); i++){
      //     cout << tupla.get(i).to_string(Angle::DEG).str() << (i==tupla.size()-1 ? "" : ", ");
      //   }
      //   cout << ">" << endl;
      // }
      // cout << endl;

      // Mat image(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));

      // for (auto point : _points){
      //     rectangle(image, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
      // }

      // my_imshow("dubin", image, true);
      // mywaitkey();
    #endif

    //Compute Dubins
    // Tuple<Tuple<Dubins<T> > > allDubins;
    this->L=DInf;
    int id=0;
    for (int j=0; j<angles.size(); j++){
      Tuple<Dubins<T> > app;
      double l=0.0;
      Tuple<Angle> angleT=angles.get(j);
      for (int i=0; i<angleT.size()-1; i++){
        Dubins<T> d=Dubins<T>(_points.get(i), _points.get(i+1), angleT.get(i), angleT.get(i+1), _kmax);
        if (d.getId()<0){
          app=Tuple<Dubins<T> > ();
          l=DInf;
          break;
        }
        app.add(d);
        l+=d.length();
      }
      
      if ((this->L)>l) {
        this->dubinses=app; 
        this->L=l;
        id=j;
      }

      // allDubins.add(app);
    }

    #ifdef DEBUG 
      Mat best_img(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
      for (auto point : _points){
        rectangle(best_img, Point(point.x()-INC/2+SHIFT, point.y()-INC/2+SHIFT), Point(point.x()+INC/2+SHIFT, point.y()+INC/2+SHIFT), Scalar(0,0,0) , -1);
      }
      for (auto dub : this->dubinses){
        dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img, SHIFT);
      }
      // my_imshow("best", best_img, true);
      // mywaitkey();
      cout << *this << endl;
    #endif

    _angles=angles.get(id);
  }

  double getLength()              { return this->L; }  
  double getKmax()                { return this->Kmax; } 
  double getSize()                { return this->dubinses.size(); }
  Tuple<Dubins<T> > getDubinses() { return this->dubinses; }
  
  Dubins<T> getDubins(int id){
    if (id<this->size()){
      return dubinses.get(id);
    }
    return Dubins<T>();
  }

  stringstream to_string() {
    stringstream out;
    out << "Total length: " << L << "\n";
    for (auto dub : dubinses){
      out << dub << endl;
    }
    return out;
  }

  friend ostream& operator<<(ostream &out, DubinsSet& data) {
    out << data.to_string().str();
    return out;
  }


};

#endif
