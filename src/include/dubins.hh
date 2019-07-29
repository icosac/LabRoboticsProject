#ifndef DUBINS_HH
#define DUBINS_HH

#include <maths.hh>
#include <utils.hh>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#if defined DEBUG && defined REALLY_DEBUG
#include <cstdio> // For sprintf
#endif

//TODO find which function is faster
#define MORE_FUNCTIONS
#define PIECE_LENGTH 2 //cm

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
  double x=_P0.x()+_L*sinc(_K*_L/2.0) * cos(_P0.angle().toRad()+_K*_L/2);
  double y=_P0.y()+_L*sinc(_K*_L/2.0) * sin(_P0.angle().toRad()+_K*_L/2);
  Angle th=Angle(_K*_L+_P0.angle().toRad(), Angle::RAD);

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
    K=_k;
    L=_l;
    Configuration2<T2> _P1 = circline(L, _P0, K);
    Curve<T2>::begin(_P0); Curve<T2>::end(_P1);
  }
#else
  DubinsArc <T2>(const Configuration2<T2> _P0,
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

  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image){
    // Mat imageMap(dimX, dimY, CV_8UC3, Scalar(255,255,255));
    for (auto point : this->splitIt(1)){
      if (point.x()>dimX || point.y()>dimY){
        double x=point.x()>dimX ? point.x() : dimX;
        double y=point.y()>dimY ? point.y() : dimY;
        Mat newMat(x, y, CV_8UC3, Scalar(255, 255, 255));
        for (double _x=0; _x<dimX; _x++){
          for (double _y=0; _y<dimY; _y++){
            rectangle(newMat, Point(_x, _y),Point(_x+inc, _y+inc), scl, -1);
          }
        }
        image=newMat;
      }
      rectangle(image, Point(point.x(), point.y()), Point(point.x()+inc, point.y()+inc), scl, -1);
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
  Tuple<double> LSL (Angle th0, Angle th1, double _kmax)
  {
    double C=th1.cos()-th0.cos();
    double S=2*_kmax+th0.sin()-th1.sin();
    
    double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
    
    if (temp1<0){
      return Tuple<double> (0);
    }
    
    double invK=1/_kmax;
    double sc_s1=Angle(atan2(C,S)-th0.get(), Angle::RAD).get()*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=Angle(th1.get()-atan2(C,S), Angle::RAD).get()*invK;
    
    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RSR (Angle th0, Angle th1, double _kmax)
  {
    double C=th0.cos()-th1.cos();
    double S=2*_kmax-th0.sin()+th1.sin();
    
    double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
    
    if (temp1<0){
      return Tuple<double> (0);
    }
    
    double invK=1/_kmax;
    double sc_s1=Angle(th0.get()-atan2(C,S), Angle::RAD).get()*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=Angle(atan2(C,S)-th1.get(), Angle::RAD).get()*invK;

    return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> LSR (Angle th0, Angle th1, double _kmax)
  {    
    double C = th0.cos()+th1.cos();
    double S=2*_kmax+th0.sin()+th1.sin();
    
    double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()+th1.sin());
    if (temp1<0){
      return Tuple<double> (0);
    }
    
    double invK=1/_kmax;
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0.get(), Angle::RAD).get()*invK;
    double sc_s3= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1.get(), Angle::RAD).get()*invK;

    return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RSL (Angle th0, Angle th1, double _kmax)
  {
    double C = th0.cos()+th1.cos();
    double S=2*_kmax-th0.sin()-th1.sin();
    
    double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()+th1.sin());
    if (temp1<0){
      return Tuple<double> (0);
    }
    
    double invK=1/_kmax;
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1= Angle(th0.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
    double sc_s3= Angle(th1.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
    
    return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> RLR (Angle th0, Angle th1, double _kmax)
  {
    double C=th0.cos()-th1.cos();
    double S=2*_kmax-th0.sin()+th1.sin();
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin()));
    
    if (fabs(temp1)-Epsi>1.0){
      return Tuple<double> (0);
    }

    if (equal(fabs(temp1), 1.0) ){
      temp1=round(temp1);
    }
    
    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(th0.get()-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th0.get()-th1.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  Tuple<double> LRL (Angle th0, Angle th1, double _kmax)
  {
    double C=th1.cos()-th0.cos();
    double S=2*_kmax+th0.sin()-th1.sin();
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin()));

    if (fabs(temp1)-Epsi>1.0){
      return Tuple<double> (0);
    }

    if (equal(fabs(temp1), 1.0) ){
      temp1=round(temp1);
    }

    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(atan2(C, S)-th0.get()+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th1.get()-th0.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
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
    Tuple<double> scaled = scaleToStandard();
    
    Angle  sc_th0     =  Angle(scaled.get(0), Angle::RAD);
    Angle  sc_th1     =  Angle(scaled.get(1), Angle::RAD); 
    double sc_Kmax    =  scaled.get(2);
    double sc_lambda  =  scaled.get(3);

    double Length = DInf;
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
        if (appL<Length){
          // first_go=false;
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
      A1=DubinsArc<T>(Curve<T>::begin(), ksigns[pidx][0]*Kmax, sc_std.get(0));
      A2=DubinsArc<T>(A1.end(), ksigns[pidx][1]*Kmax, sc_std.get(1));
      A3=DubinsArc<T>(A2.end(), ksigns[pidx][2]*Kmax, sc_std.get(2));
#else
      double L = sc_std.get(0);
      double K = ksigns[pidx][0];
      Configuration2<double> _P1 = circline(L, Curve<T>::begin(), K);
      A1=DubinsArc<T>(Curve<T>::begin(), _P1, K, L);
      
      L = sc_std.get(1); K = ksigns[pidx][1];
      _P1 = circline(L, A1.begin(), K);
      A2=DubinsArc<T>(A1.begin(), _P1, K, L);
      
      L = sc_std.get(2); K = ksigns[pidx][2];
      _P1 = circline(L, A2.begin(), K);
      A3=DubinsArc<T>(A2.begin(), _P1, K, L);
#endif
      L=A1.length()+A2.length()+A3.length(); //Save total length of Dubins curve

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

  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image){
    A1.draw(dimX, dimY, inc, scl, image);
    A2.draw(dimX, dimY, inc, scl, image);
    A3.draw(dimX, dimY, inc, scl, image);
  }

};

//TODO find non recursive approach
/*! \brief Compute the arrangements.
 */
Tuple<Tuple<Angle> > t;
void disp ( Tuple<Angle>& z,    ///<Vector to use
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
      disp(z, id-1, N, inc, startPos);
      // a+=inc;
      z.set(id, z.get(id)+inc);
      if (i==N-1)
        disp(z, id-1, N, inc, startPos);
    }
    z.set(id, start);
  }
}

/*!\brief Given a set of point, compute the shortest set of Dubins that allows to go from start to end through all points.
 *
 */
template <class T>
class DubinsSet {
private: 
  Tuple<Dubins<T> > dubinses;
  double Kmax, L;

  DubinsSet(Tuple<Dubins<T> > _dubinses,
            double _kmax=KMAX){
    dubinses=_dubinses;
    Kmax=_kmax;
    for (Dubins<T> dub : _dubinses){
      L+=dub.length();
    } 
  }

  DubinsSet(Tuple<Configuration2<T> > _confs,
            double _kmax=KMAX){
    for (int i=0; i<_confs.size()-1; i++){
      Dubins<T> dub=Dubins<T>(_confs.get(i), _confs.get(i+1));
      dubinses.add(dub);
      L+=dub.length();
    }
    Kmax=_kmax;
  }

  DubinsSet(Configuration2<T> start, 
            Configuration2<T> end,
            Tuple<Point2<T> > _points,
            double _kmax=KMAX){
  }

  DubinsSet(Tuple<Point2<T> > _points,
            double _kmax=KMAX){
    // uint size=_points.size();
    // for (uint i=0; i<size-1; i++){ //Cycle through all pair of points
    // }
  }
};

#endif
