#ifndef DUBINS_HH
#define DUBINS_HH

#include "maths.hh"
#include "utils.hh"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#if defined DEBUG && defined REALLY_DEBUG
#include <cstdio> // For sprintf
#endif

#define MORE_FUNCTIONS

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

static Configuration2<double> circline (double _L,
                                        Configuration2<double> _P0,
                                        double _K){
  double x=_P0.x()+_L*sinc(_K*_L/2.0) * cos(_P0.angle().get()+_K*_L/2);
  double y=_P0.y()+_L*sinc(_K*_L/2.0) * sin(_P0.angle().get()+_K*_L/2);
  Angle th=_P0.angle()+Angle(_K, Angle::RAD)*_L;
  return Configuration2<double>(x, y, th);
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
  DubinsArc ( const Configuration2<T2> _P0,
              const T1 _k,
              const T1 _l) : Curve<T2>() {
    K=_k;
    L=_l;
    Configuration2<T2> _P1 = circline(L, _P0, K);
    Curve<T2>::begin(_P0); Curve<T2>::end(_P1);
  }
#else
  DubinsArc (const Configuration2<T2> _P0,
             const Configuration2<T2> _P1,
             const T1 _k,
             const T1 _l) : Curve<T2>(_P0, _P1) {
    K=_k;
    L=_l;
    // cout << "_P0: " << _P0 << endl;
    // cout << "begin: " << Curve<T2>::begin() << endl;
  }
#endif

  T1 getK   () const { return K; }
  T1 lenght () const { return L; }

  stringstream to_string() const {
    stringstream out;
    out << "begin: " << Curve<T2>::begin();
    out << ", end: " << Curve<T2>::end();
    out << ", K: " << getK();
    return out;
  }
  
  friend ostream& operator<<(ostream &out, const DubinsArc& data) {
    out << data.to_string().str();
    return out;
  }

};


#define KMAX 1.0
template<class T>
class Dubins : protected Curve<T>
{
private:
  double Kmax, L;
  DubinsArc<> A1, A2, A3;

  using Curve<T>::Curve;
  // using DubinsArc<>::DubinsArc;
public:
  Dubins () : Kmax(KMAX), Curve<T>() {
    A1=DubinsArc<>();
    A2=DubinsArc<>();
    A3=DubinsArc<>();
  }

  Dubins (const Configuration2<T> _P0,
          const Configuration2<T> _P1,
          const double _K=KMAX) :
  Curve<T>(_P0, _P1), Kmax(_K) {
    if (shortest_path()<0){
      A1=DubinsArc<>();
      A2=DubinsArc<>();
      A3=DubinsArc<>();
    }
  }

  Dubins (const Point2<T> _P0,
          const Point2<T> _P1,
          const Angle _th0,
          const Angle _th1,
          const double _K=KMAX) :
  Curve<T>(_P0, _P1, _th0, _th1), Kmax(_K) {
    if (shortest_path()<0){
      A1=DubinsArc<>();
      A2=DubinsArc<>();
      A3=DubinsArc<>();
    }
  }

  Dubins (const T x0, const T y0,
          const Angle _th0,
          const T x1, const T y1,
          const Angle _th1,
          const double _K=KMAX) :
  Curve<T>(x0, y0, _th0, x1, y1, _th1), Kmax(_K) {
    if (shortest_path()<0){
      A1=DubinsArc<>();
      A2=DubinsArc<>();
      A3=DubinsArc<>();
    }
  }

  double getKMax  () const { return Kmax; }
  double lenght   () const { return L; }

  DubinsArc<> getA1() const { return A1; }
  DubinsArc<> getA2() const { return A2; }
  DubinsArc<> getA3() const { return A3; }

  Tuple<double> LSL (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th1.cos()-th0.cos();
    double S = 2*_kmax + th0.sin()-th1.sin();

    Angle temp1 (atan2(C,S), Angle::RAD);
    double sc_s1 = invK*(temp1-th0).get();

    double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
    if (temp2<0){
      return Tuple<double>(0);
      TOFILE("data/test/CC_LSL.test", "0\n");
    }
    double sc_s2 = invK * sqrt(temp2);
    double sc_s3 = invK*(th1-temp1).get();

#if defined DEBUG  && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_LSL.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RSR (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th0.cos()-th1.cos();
    double S = 2*_kmax - th0.sin()+th1.sin();

    Angle temp1 (atan2(C,S), Angle::RAD);
    double sc_s1 = invK*(th0-temp1).get();

    double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
    if (temp2<0){
      TOFILE("data/test/CC_RSR.test", "0\n");
      return Tuple<double>(0);
    }
    double sc_s2 = invK * sqrt(temp2);
    double sc_s3 = invK*(temp1-th1).get();

#if defined DEBUG && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_RSR.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> LSR (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th0.cos()+th1.cos();
    double S = 2*_kmax + th0.sin()+th1.sin();

    Angle temp1 (atan2(-C,S), Angle::RAD);

    double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax * (th0.sin() + th1.sin());
    if (temp2<0){
      TOFILE("data/test/CC_LSR.test", "0\n");
      return Tuple<double>(0);
    }
    double sc_s2 = invK * sqrt(temp2);
    Angle temp3 = Angle(-atan2(-2, sc_s2*_kmax), Angle::RAD);
    double sc_s1 = invK * (temp1+temp3-th0).get();
    double sc_s3 = invK * (temp1+temp3-th1).get();

#if defined DEBUG && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_LSR.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RSL (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th0.cos()+th1.cos();
    double S = 2*_kmax - th0.sin()-th1.sin();

    Angle temp1 (atan2(C,S), Angle::RAD);

    double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax*(th0.sin() + th1.sin());
    if (temp2<0){
      TOFILE("data/test/CC_RSL.test", "0\n");
      return Tuple<double>(0);
    }
    double sc_s2 = invK * sqrt(temp2);
    Angle temp3 = Angle(-atan2(-2, sc_s2*_kmax), Angle::RAD);
    double sc_s1 = invK * ((th0-temp1+temp3)).get();
    double sc_s3 = invK * ((th1-temp1+temp3)).get();

#if defined DEBUG && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_RSL.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RLR (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th0.cos()-th1.cos();
    double S = 2*_kmax - th0.sin()+th1.sin();

    Angle temp1 (atan2(C,S), Angle::RAD);
    double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() + 4*_kmax*(th0.sin()-th1.sin()));
    if (std::abs(temp2)>1){
      TOFILE("data/test/CC_RLR.test", "0\n");
      return Tuple<double>(0);
    }

    double sc_s2 = invK*(Angle(2*M_PI-acos(temp2), Angle::RAD).get());
    double sc_s1 = invK*(th0-temp1+Angle(sc_s2*(0.5*_kmax), Angle::RAD)).get();
    double sc_s3 = invK*(th1-th0+Angle(sc_s2-sc_s1, Angle::RAD)*_kmax).get();

#if defined DEBUG && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_RLR.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> LRL (Angle th0, Angle th1, double _kmax)
  {
    double invK = 1/_kmax;
    double C = th1.cos()-th0.cos();
    double S = 2*_kmax + th0.sin()-th1.sin();

    Angle temp1 (atan2(C,S), Angle::RAD);
    double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() - 4*_kmax*(th0.sin()-th1.sin()));

    if (std::abs(temp2)>1){
      TOFILE("data/test/CC_LRL.test", "0\n");
      return Tuple<double>(0);
    }
    double sc_s2 = invK*(Angle(2*M_PI-acos(temp2), Angle::RAD)).get();
    double sc_s1 = invK*(temp1-th0+Angle(sc_s2*(0.5*_kmax), Angle::RAD)).get();
    double sc_s3 = invK*(th1-th0+Angle(sc_s2-sc_s1, Angle::RAD)*_kmax).get();

#if defined DEBUG && defined REALLY_DEBUG
    char output [256];
    sprintf(output, "%f, %f, %f", sc_s1, sc_s2, sc_s3);
    TOFILE("data/test/CC_LRL.test", output);
#endif
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> scaleToStandard ()
  {
    double dx=Curve<T>::end().x() - Curve<T>::begin().x();
    double dy=Curve<T>::end().y() - Curve<T>::begin().y();

    double _phi=atan2(dy, dx);
    Angle phi= Angle(_phi, Angle::RAD);

    double lambda=sqrt(pow2(dx)+pow2(dy));

    double C = dx /lambda;
    double S = dy /lambda;

    lambda /= 2;

    Angle sc_th0 = Angle (Curve<T>::begin().angle().get()-phi.get(), Angle::RAD);
    Angle sc_th1 = Angle (Curve<T>::end().angle().get()-phi.get(), Angle::RAD);
    double sc_Kmax = Kmax*lambda;

    return Tuple<double> (4, sc_th0.get(), sc_th1.get(), sc_Kmax, lambda);
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
    Tuple<double> scaled = scaleToStandard();
    
    Angle  sc_th0     =  Angle(scaled.get(0), Angle::RAD);
    Angle  sc_th1     =  Angle(scaled.get(1), Angle::RAD); 
    double sc_Kmax    =  scaled.get(2);
    double sc_lambda  =  scaled.get(3);

    //TODO Missing ksigns matrix
    double Length = -1.0;
    double sc_s1  = 0.0;
    double sc_s2  = 0.0;
    double sc_s3  = 0.0;
    bool first_go = true;

    std::vector<Tuple<double> > res;
    res.push_back(LSL(sc_th0, sc_th1, sc_Kmax));
    res.push_back(RSR(sc_th0, sc_th1, sc_Kmax));
    res.push_back(LSR(sc_th0, sc_th1, sc_Kmax));
    res.push_back(RSL(sc_th0, sc_th1, sc_Kmax));
    res.push_back(RLR(sc_th0, sc_th1, sc_Kmax));
    res.push_back(LRL(sc_th0, sc_th1, sc_Kmax));

    int i=0;
    for (auto value : res){
      if (value.size()>0){
        double appL=value.get(0)+value.get(1)+value.get(2);
        if (appL<Length || first_go){
          first_go=false;
          Length = appL;
          sc_s1=value.get(0);
          sc_s2=value.get(1);
          sc_s3=value.get(2);
          pidx=i;
        }
      }
      i++;
    }

    if (pidx>=0){
      Tuple<double> sc_std = scaleFromStandard(sc_lambda, sc_s1, sc_s2, sc_s3);
      vector<vector<int> > ksigns ={
        { 1,  0,  1}, // LSL
        {-1,  0, -1}, // RSR
        { 1,  0, -1}, // LSR
        {-1,  0,  1}, // RSL
        {-1,  1, -1}, // RLR
        { 1, -1,  1}  // LRL
      };

#ifdef MORE_FUNCTIONS
      A1=DubinsArc<>(Curve<T>::begin(), ksigns[pidx][0], sc_std.get(0));
      A2=DubinsArc<>(A1.end(), ksigns[pidx][1], sc_std.get(1));
      A3=DubinsArc<>(A2.end(), ksigns[pidx][2], sc_std.get(2));
#else
      double L = sc_std.get(0);
      double K = ksigns[pidx][0];
      Configuration2<double> _P1 = circline(L, Curve<T>::begin(), K);
      A1=DubinsArc<>(Curve<T>::begin(), _P1, K, L);
      
      L = sc_std.get(1); K = ksigns[pidx][1];
      _P1 = circline(L, A1.begin(), K);
      A2=DubinsArc<>(A1.begin(), _P1, K, L);
      
      L = sc_std.get(2); K = ksigns[pidx][2];
      _P1 = circline(L, A2.begin(), K);
      A3=DubinsArc<>(A2.begin(), _P1, K, L);
#endif
      L=A1.lenght()+A2.lenght()+A3.lenght(); //Save total length of Dubins curve

      bool check_ = check(sc_s1, ksigns[pidx][0]*sc_Kmax,
                          sc_s2, ksigns[pidx][1]*sc_Kmax,
                          sc_s3, ksigns[pidx][2]*sc_Kmax,
                          sc_th0, // Curve<T>::begin().angle(),
                          sc_th1  // Curve<T>::end().angle()
                        );
      if (!check_)
        pidx=-1.0;
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
              Angle th1) const {
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

};

#endif
