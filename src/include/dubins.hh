#ifndef DUBINS_HH
#define DUBINS_HH

// #include <utils.hh>
#include <maths.hh>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>

#if defined DEBUG && defined REALLY_DEBUG
#include <cstdio> // For sprintf
#endif

#ifdef DEBUG
#define D_INC 5
#define D_SHIFT 100
#define D_DIMX 200+D_SHIFT
#define D_DIMY 500+D_SHIFT
#endif

#define PIECE_LENGTH 2 //mm
#define PREC 100000

using namespace std;

/*!
 * Class that defines a general curve. It just containes a start `Configuration2` and an end `Configuration2`.
 * \tparam T The type of the `Configuration2`s
 */
template <class T>
class Curve {
protected:
  Configuration2<T> P0; ///< Start `Configuration2`.
  Configuration2<T> P1; ///< End `Configuration2`.

public:
  Curve () : P0(), P1() {} ///<Default plain constructor which creates two plain `Configuration2`s

  /*!
   * Constructor that takes two `Configuration2`s and stores them.
   * \param[in] _P0 Start `Configuration2`.
   * \param[in] _P1 End `Configuration2`.
   */
  Curve (const Configuration2<T> _P0,
         const Configuration2<T> _P1) :
  P0(_P0), P1(_P1) {}

  /*!
   * Constructor that takes two `Point2`s and two `Angle`s and stores them as `Configuration2`s.
   * \param[in] _P0 Start `Point2`.
   * \param[in] _P1 End `Point2`.
   * \param[in] _th0 Starting `Angle`
   * \param[in] _th1 Ending `Angle`
   */
  Curve (const Point2<T> _P0,
         const Point2<T> _P1,
         const Angle _th0,
         const Angle _th1) :
  P0(_P0, _th0), P1(_P1, _th1) {}

  /*!
   * Constructor that takes the bare coordinates of two points and their `Angle`s and stores them as `Configuration2`s.
   * \param[in] x0 Start abscissa coordinate.
   * \param[in] y0 Start ordinate coordinate.
   * \param[in] _th0 Start `Angle`.
   * \param[in] x1 End abscissa coordinate.
   * \param[in] y1 End ordinate coordinate.
   * \param[in] _th1 End `Angle`.
   */
  Curve (const T x0, const T y0,
         const Angle _th0,
         const T x1, const T y1,
         const Angle _th1) :
  P0(x0, y0, _th0), P1(x1, y1, _th1) {}

  Configuration2<T> begin()  const { return P0; } ///< Returns the starting `Configuration2` of the `Curve`.
  Configuration2<T> end()    const { return P1; } ///< Returns the ending `Configuration2` of the `Curve`.

  /*!
   * Function that stores the starting `Configuration2`.
   * \param[in] _P0 Starting `Configuration2`.
   */
  void begin(Configuration2<T> _P0){
    P0=_P0;
  }
  /*!
   * Function that stores the ending `Configuration2`.
   * \param[in] _P0 Ending `Configuration2`.
   */
  void end (Configuration2<T> _P1){
    P1=_P1;
  }

  /*! This function create a strinstream object containing infos about the `Curve`.
      \returns A string stream.
  */
  stringstream to_string() const{
    stringstream out;
    out << "begin: " << begin();
    out << ", end: " << end();
    return out;
  }

  /*!
   * This function overload the << operator so to print with `std::cout` the values of the `Curve`.
   * \param[in] out The out stream.
   * \param[in] data The `Curve` to print.
   * \returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const Curve& data) {
    out << data.to_string().str();
    return out;
  }
};

/*!
 * Compute the sinc of the function defined as: \f[
 * sinc(t)=\frac{sin(t)}{t}\quad t\neq 0
 * 1\quad t=0
 * \f]
 * \param[in] t The value of the angle to be used.
 * \return The result of the previous formula.
 */
inline double sinc(double t) {
  if (equal(t, 0.0))
    return 1 - pow2(t)/6 * (1 - pow2(t)/20);
  else
    return sin(t)/t;
}

/*!
 * Computes an arrival point from an initial configuration through an arc of length _L and curvature _K.
 * \param[in] _L The length of the arch.
 * \param[in] _P0 The starting `Configuration2` of the arc.
 * \param[in] _K The curvature of the arc.
 * \return The ending `Configuration2` of the arc.
 */
Configuration2<double> circline(double _L,
                                Configuration2<double> _P0,
                                double _K);

//TODO document
template<class T>
bool is_on_circarc( Point2<T> p0, 
                    Point2<T> pi, 
                    Point2<T> pf,
                    Point2<T> p
                  ){
  //Compute circonference with Cramer
  double d0=-((pow2(p0.x())+pow2(p0.y())));
  double di=-((pow2(pi.x())+pow2(pi.y())));
  double df=-((pow2(pf.x())+pow2(pf.y())));
  //Compute determinants
  double D = (p0.x()*pi.y())+(p0.y()*pf.x())+(pi.x()*pf.y()) - ( p0.x()*pf.y()+p0.y()*pi.x()+(pi.y()*pf.x()) );
  double Da= (p0.x()*d0)+(p0.y()*df)+(pf.y()*di) - ( d0*pf.y()+di*p0.y()+df*pi.y() );
  double Db= (p0.x()*di)+(d0*pf.x())+(pi.x()*df) - ( df*p0.x()+ d0*pi.x()+di*pf.x() );
  double Dc= (p0.x()*pi.y()*df)+(p0.y()*di*pf.x())+(d0*pi.x()*pf.y()) - ( p0.x()*di*pf.y()+p0.y()*pi.x()*df+d0*pi.y()*pf.x() );
  //Compute circle's parameters
  double a=Da/D;
  double b=Db/D;
  double c=Dc/D;

  bool ok=true;

  if ( equal((pow2(p.x())+pow2(p.y())+a*p.x()+b*p.y()+c), 0.0) ){ //Check if point is on circonference
    Point2<double> center (-a/2.0, -b/2.0); //Compute center

    //Compute angles of extremities and point of intereset p. Mind that center.th(p0) is 0 only if p0 is on the right.
    Angle th0=center.th(p0);
    Angle thf=center.th(pf);
    Angle th=center.th(p);
  
    if (th>=th0 && th<=thf){ //Point is not in the arc.
      ok=false;
    }
  }
  else {
    ok=false;
  }
  return ok;
}

/*!
 * \brief Class to store a maneuver of Dubins. It inherits from `Curve`.
 * Since each Dubins is formed of atmost 3 maneuvers, this class is meant to store one of this maneuver, which can be L, R or S respectively Left, Right, Straight.
 * \tparam T1 The type of Length and Curvature.
 * \tparam T2 The type of the class `Curve`.
 */
template <class T1=double, class T2=double>
class DubinsArc : public Curve<T2>
{
private:
  T1 L; ///< Length of the arc.
  T1 K; ///< Curvature of the arc.

  using Curve<T2>::Curve;
public:

  /*!
   * Plain constructor of `DubinsArc` that sets L and K to 0 and creates a plain `Curve`.
   */
  DubinsArc () : L(0), K(0), Curve<T2>() {}

  /*!
   * Creates a new `DubinsArc` given a start `Configuration2`, the curvature and the length of the arc calling `circline()`.
   * \param[in] _P0 The starting `Configuration2`.
   * \param[in] _k The curvature of the `DubinsArc`.
   * \param[in] _l The length of the `DubinsArc`.
   */
  DubinsArc(const Configuration2<T2> _P0,
            const T1 _k,
            const T1 _l) : Curve<T2>() {
    K=_k;
    L=_l;

    Configuration2<T2> _P1 = circline(L, _P0, K);
    Curve<T2>::begin(_P0); Curve<T2>::end(_P1);
  }

  T1 getK   () const { return K; } ///< Returns the curvature of the arc.
  T1 length () const { return L; } ///< Returns the length of the arc.

  //TODO Add last point of curve
  /*!
   * \brief Splits the `DubinsArc` in pieces of _L length.
   * This function starts from the begining of the arc and computes n new arcs through the `circline()` function using the curvature of the `DubinsArc` and _L as the length.
   * \param[in] _L The length that each points should have.
   * \return A `Tuple` of `Configuration2`s representing the points along the arc.
   */
  Tuple<Configuration2<T2> > splitIt (double _L=PIECE_LENGTH){
    Tuple<Configuration2<T2> > ret;
    Configuration2<T2> _old=Curve<T2>::begin();
    double sum=0.0;

    while( this->length()>sum+_L ){
      Configuration2<T2> _new=circline(_L, _old, getK());
      ret.add(_new);
      _old=_new; //Maybeeeee using pointers can improve performance?
      sum+=_L;
    }

    ret.add(Curve<T2>::end());
    return ret;
  }

  /*! This function create a strinstream object containing infos about the `DubinsArc`.
      \returns A string stream.
  */
  stringstream to_string() const {
    stringstream out;
    out << "begin: " << Curve<T2>::begin();
    out << ", end: " << Curve<T2>::end();
    out << ", K: " << getK();
    out << ", l: " << length();
    return out;
  }

  /*!
   * This function overload the << operator so to print with `std::cout` the values of the `DubinsArc`, that is `Curve` values more the length and the curvature.
   * \param[in] out The out stream.
   * \param[in] data The `DubinsArc` to print.
   * \returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const DubinsArc& data) {
    out << data.to_string().str();
    return out;
  }

  /*!
   * This function draws the `DubinsArc`.
   * \param[in] dimX The dimension X of the Mat.
   * \param[in] dimY The dimension Y of the Mat.
   * \param[in] inc The value to scale each point.
   * \param[in] scl The Scalar that defines the color to use.
   * \param[in] image The Mat where to draw the points.
   * \param[in] D_SHIFT The value to use to shift the points to make them stay inside the matrix.
   */
  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image, double D_SHIFT){
    // Mat imageMap(dimX, dimY, CV_8UC3, Scalar(255,255,255));
    for (auto point : this->splitIt(1)){
      if (point.x()>dimX || point.y()>dimY){
        double x=point.x()>dimX ? point.x() : dimX;
        double y=point.y()>dimY ? point.y() : dimY;
        Mat newMat(x, y, CV_8UC3, Scalar(255, 255, 255));
        for (double _x=0; _x<dimX; _x++){
          for (double _y=0; _y<dimY; _y++){
            rectangle(newMat, Point(_x+D_SHIFT, _y+D_SHIFT),Point(_x+inc+D_SHIFT, _y+inc+D_SHIFT), scl, -1);
          }
        }
        image=newMat;
      }
      rectangle(image, Point(point.x()+D_SHIFT, point.y()+D_SHIFT), Point(point.x()+inc+D_SHIFT, point.y()+inc+D_SHIFT), scl, -1);
    }
  }

  bool is_on_dubinsArc(Configuration2<T2> C){ 
    bool ok=false;
    if (!equal(this->length(), 0.0)){
      if (equal(this->getK(), 0.0)){ //Check if on line
        if(Curve<T2>::begin().angle()==C.angle()){
          if (Curve<T2>::begin().point().th(C.point())==C.angle()) {
            ok=true;
          }
        }
      }
      else {
        Configuration2<T2> intermediate=circline(this->length()-this->length()/100.0, Curve<T2>::begin(), this->getK());
        ok=is_on_circarc(Curve<T2>::begin().point(), intermediate.point(), Curve<T2>::end().point(), C.point());
      }
    }
    return ok;
  }
};


#define KMAX 0.01
/*!
 * \brief Class to store a Dubins curve.
 * This class inherits from `Curve` and is composed of three `DubinsArc`.
 * \tparam T The type of the classes `Curve` and `DubinsArc`.
 */
template<class T>
class Dubins : protected Curve<T>
{
private:
  double Kmax;             ///< The curvature of the Dubins.
  double L;                ///< The length of the curve.
  int pid=-1;              ///< An ID that indicates which set of maneuver composes the Dubins.
  DubinsArc<T> A1, A2, A3; ///< The 3 arcs that compose the Dubins.

  using Curve<T>::Curve;
public:
  /*!
   * Plain constructor for Dubins that calls the plain constructor of `Curve` and `DubinsArc`.
   */
  Dubins () : Kmax(KMAX), Curve<T>(), L(DInf) {
    A1=DubinsArc<T>();
    A2=DubinsArc<T>();
    A3=DubinsArc<T>();
  }

  /*!
   * Constructor that takes an initial and a final `Configuration2`, a curvature and compute the Dubins that connect the two configurations.
   * \param[in] _P0 Initial `Configuration2`.
   * \param[in] _P1 Final `Configuration2`.
   * \param[in] _K Curvature.
   */
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

  /*!
   * Constructor that takes an initial and a final `Point2`, the two respectively `Angle`s and the curvature and computes the Dubins.
   * \param[in] _P0 Initial `Point2`.
   * \param[in] _P1 Final `Point2`.
   * \param[in] _th0 Initial `Angle`
   * \param[in] _th1 Final `Angle`
   * \param[in] _K Curvature.
   */
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

  /*!
   * Constructor that takes the initial and final coordinates, the respective `Angle`s and the curvature and compute a Dubins.
   * \param[in] x0 Initial abscissa coordinate.
   * \param[in] y0 Initial ordinate coordinate.
   * \param[in] _th0 Initial `Angle`.
   * \param[in] x1 Final abscissa coordinate.
   * \param[in] y1 Final ordinate coordinate.
   * \param[in] _th1 Final `Angle`.
   * \param[in] _K Curvature of the curve.
   */
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

  double getKmax  () const { return Kmax; } ///< Returns the maximum curvature of the Dubins.
  double length   () const { return L; }    ///< Returns the length of the Dubins.
  double getId    ()  { return pid; }       ///< Returns the id of the Dubins, that is the set of three maneuvers that creates the curve.

  Configuration2<T> begin () const { return Curve<T>::begin(); }
  Configuration2<T> end () const { return Curve<T>::end(); }

  DubinsArc<T> getA1() const { return A1; } ///< Returns the first `DubinsArc`.
  DubinsArc<T> getA2() const { return A2; } ///< Returns the second `DubinsArc`.
  DubinsArc<T> getA3() const { return A3; } ///< Returns the third `DubinsArc`.

  /*!
   * Function to compute the set of maneuvers Left Straight Left.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* LSL (double th0, double th1, double _kmax)
  {
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
    

    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1.get(), sc_s2, sc_s3.get());
  }

  /*!
   * Function to compute the set of maneuvers Right Straight Right.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* RSR (double th0, double th1, double _kmax)
  {
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
    

    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    
    // return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
  }

  /*!
   * Function to compute the set of maneuvers Left Straight Right.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* LSR (double th0, double th1, double _kmax)
  {    
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


    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  /*!
   * Function to compute the set of maneuvers Right Straight Left.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* RSL (double th0, double th1, double _kmax)
  {
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
    
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  /*!
   * Function to compute the set of maneuvers Right Left Right.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* RLR (double th0, double th1, double _kmax)
  {
    double C=cos(th0)-cos(th1);
    double S=2*_kmax-sin(th0)+sin(th1);
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));
    
    if (fabs(temp1)-Epsi>1.0){
      // return Tuple<double> (0);
      return nullptr;
    }

    if (equal(fabs(temp1), 1.0) ){
      temp1=round(temp1);
    }
    
    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(th0-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th0-th1+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;
    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  /*!
   * Function to compute the set of maneuvers Left Right Left.
   * \param[in] th0 The initial angle standardized.
   * \param[in] th1 The final angle standardized.
   * \param[in] _kmax The maximum curvature.
   * \return An array of dimension 3 containing the length of the 3 maneuvers.
   */
  double* LRL (double th0, double th1, double _kmax)
  {
    double C=cos(th1)-cos(th0);
    double S=2*_kmax+sin(th0)-sin(th1);
    
    double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

    if (fabs(temp1)-Epsi>1.0){
      // return Tuple<double> (0);
      return nullptr;
    }

    if (equal(fabs(temp1), 1.0) ){
      temp1=round(temp1);
    }

    double invK=1/_kmax;
    double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
    double sc_s1 = Angle(atan2(C, S)-th0+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
    double sc_s3 = Angle(th1-th0+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
    
    
    double* ret=new double[3];
    ret[0]=sc_s1;
    ret[1]=sc_s2;
    ret[2]=sc_s3;

    return ret;

    // return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
  }

  /*!
   * \brief Function to compute standardize the parameters.
   * This function computes the initial and final angles as if the reference system is P0(-1,0), P1(0,1). This allows to simplify the calculations to find the best set of maneuvers.
   * \return A `Tuple` of `duoble` containing the standardised initial and final angle, the new curvature and the parameter lambda that allows to compute the real dimension lengths.
   */
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

  /*!
   * \brief This function computes the shortest path for the Dubins constructed.
   * First the values are scaled. Then the six sets of maneuvers are computed and their lengths are stored. Once the set that gives the Dubins with the minimum length is found, the lengths are rescaled and the `DubinsArc` are created. In the process length is also computed.
   * \return The id of the set of maneuvers.
   */
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

    Tuple<double* > res;
    res.add(LSL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RSR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(LSR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RSL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(RLR(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));
    res.add(LRL(sc_th0.toRad(), sc_th1.toRad(), sc_Kmax));

    int i=0; 
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
      Tuple<double> sc_std = scaleFromStandard(sc_lambda, sc_s1, sc_s2, sc_s3);
      vector<vector<int> > ksigns ={
        { 1,  0,  1}, // LSL
        {-1,  0, -1}, // RSR
        { 1,  0, -1}, // LSR
        {-1,  0,  1}, // RSL
        {-1,  1, -1}, // RLR
        { 1, -1,  1}  // LRL
      };


      A1=DubinsArc<T>(Curve<T>::begin(), ksigns[pidx][0]*Kmax, sc_std.get(0));
      A2=DubinsArc<T>(A1.end(), ksigns[pidx][1]*Kmax, sc_std.get(1));
      A3=DubinsArc<T>(A2.end(), ksigns[pidx][2]*Kmax, sc_std.get(2));

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

  /*!
   * Function that checks that the values got in `shortest_path()` are right.
   * \param[in] s1 Length for the first `DubinsArc`.
   * \param[in] k0 Curvature for the first `DubinsArc`.
   * \param[in] s2 Length for the second `DubinsArc`.
   * \param[in] k1 Curvature for the second `DubinsArc`.
   * \param[in] s3 Length for the third `DubinsArc`.
   * \param[in] k2 Curvature for the third `DubinsArc`.
   * \param[in] th0 Initial angles (standardised).
   * \param[in] th1 Final angles (standardised).
   * \return `true` if the values where correct, `false` otherwise.
   */
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

  /*!
   * Normalize an angular difference \f$(-\pi, \pi]\f$.
   * \param[in] ang The value of the angle to be normalized.
   * \return The normalized angle.
  */
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
  /*!
   * Function to split a Dubins in points.
   * \param[in] _arch If defined returns only the points for a single `DubinsArc`.
   * \param[in] _L The distance from one point to another.
   * \return A `Tuple` containing three `Tuple` of `Point2` (one for each arc) containing the computed points.
   */
  Tuple<Tuple<Configuration2<double> > > splitIt (double _L=PIECE_LENGTH, 
                                                  int _arch=0){
    Tuple<Tuple<Configuration2<double> > > v;
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
    }
    return v;
  }
  /*! This function create a strinstream object containing infos about the `Dubins`.
    \returns A string stream.
  */
  stringstream to_string() const {
    stringstream out;
    out << "A1: " << getA1() << endl;
    out << "A2: " << getA2() << endl;
    out << "A3: " << getA3();
    return out;
  }

  /*!
   * This function overload the << operator so to print with `std::cout` the values of the `Dubins`, that is printing the 3 `DubinsArcs`.
   * \param[in] out The out stream.
   * \param[in] data The `Dubins` to print.
   * \returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const Dubins& data) {
    out << data.to_string().str();
    return out;
  }

  /*!
   * Function to draw the `Dubins`.
   * \param[in] dimX The dimension X of the Mat.
   * \param[in] dimY The dimension Y of the Mat.
   * \param[in] inc The value to scale each point.
   * \param[in] scl The Scalar that defines the color to use.
   * \param[in] image The Mat where to draw the points.
   * \param[in] D_SHIFT The value to use to shift the points to make them stay inside the matrix.
   */
  void draw(double dimX, double dimY, double inc, Scalar scl, Mat& image, double D_SHIFT=0){
    A1.draw(dimX, dimY, inc, scl, image, D_SHIFT);
    A2.draw(dimX, dimY, inc, scl, image, D_SHIFT);
    A3.draw(dimX, dimY, inc, scl, image, D_SHIFT);
  }

  bool is_on_dubins (Configuration2<T> C){
    return (A1.is_on_dubinsArc(C) || A2.is_on_dubinsArc(C) || A3.is_on_dubinsArc(C));
  }

};

/*!
 * \brief Convert a value in base 10 to base `base` in a `Tuple`.
 * To each value an inc is muiltiplied and the initial `Angle` is added.
 * \param[in] z A `Tuple` containing all the initial `Angle`s.
 * \param[in] n The value to be converted.
 * \param[in] base The base.
 * \param[in] inc The increment.
 * \param[in] startPos The starting position of the `Tuple` of `Angle`s.
 * \param[in] endPos The ending position of the `Tuple` of `Angle`s.
 * \return A vector containing the digits of the number converted to the specified base.
 */
Tuple<Angle> toBase(Tuple<Angle> z, int n, int base, const Angle& inc, int startPos, int endPos);

/*!
 * \brief Compute the arrangements.
 * Since each arrangement can be computed as \f$n_{parts}\f$, where each values is then multiplied for the increment and is added to the initial values.
 * \param[out] t A `Tuple` containing all the `Tuple`s containing the `Angle`s.
 * \param[in] z A `Tuple` containing all the initial `Angle`s.
 * \param[in] N The number of iterations. Each iteration is going to be converted in base parts.
 * \param[in] inc The increment to give each initial `Angle`.
 * \param[in] startPos The initial position to consider in `Tuple`.
 * \param[in] endPos The final position to consider in `Tuple`.
 */
void disp(Tuple<Tuple<Angle> >& t,
          Tuple<Angle>& z,    //Vector to use
          int N,              //Number of time to "iterate"
          const Angle& inc,   //Incrementation
          int startPos=0, 
          int endPos=0);

/*!
 * \brief Given a set of point, compute the shortest set of Dubins that allows to go from start to end through all points.
 * \tparam T Type for class `Dubins`.
 */
template <class T>
class DubinsSet {
private: 
  Tuple<Dubins<T> > dubinses; ///< `Tuple` of `Dubins` containing all the computed `Dubins`.
  double Kmax;                ///< Maximum value for curvature.
  double L=DInf;              ///< Length of all `DubinsSet`.
public:
  DubinsSet() : dubinses(), Kmax(0.0), L(DInf) {}
  /*!
   * Constructor that given a `Tuple` of `Dubins` computes stores all of them.
   * \param[in] _dubinses The `Tuple` of `Dubins`.
   * \param[in] _kmax The maximum curvature.
   */
  DubinsSet(Tuple<Dubins<T> > _dubinses,
            double _kmax=KMAX)
  {
    cout << "Creating DubinsSet from tuple " << endl << flush;
    this->dubinses=_dubinses;
    this->Kmax=_kmax;
    for (Dubins<T> dub : this->dubinses){
      this->L+=dub.length();
    } 
  }

  /*!
   * Constructor that takes a `Tuple` of `Configuration2`s and computes the `Dubins` between them.
   * \param[in] _confs The `Tuple` of `Configuration2`s.
   * \param[in] _kmax The maximum curvature to be used.
   */
  DubinsSet(Tuple<Configuration2<T> > _confs,
            double _kmax=KMAX)
  {
    for (int i=0; i<_confs.size()-1; i++){
      Dubins<T> dub=Dubins<T>(_confs.get(i), _confs.get(i+1));
      this->dubinses.add(dub);
      this->L+=dub.length();
    }
    this->Kmax=_kmax;
  }

  /*!
   * \brief Constructor that given a start `Configuration2`, an end `Configuration2` and a `Tuple` of `Point2`, computes the best path from start to end through all points by brute forcing all possible angles.
   * Since this approach is based on a brute force algorithm, it's best not to use this on too many points.
   * \param[in] start `Configuration2` of start.
   * \param[in] end `Configuration2` of end.
   * \param[in] _points `Tuple` of `Point2` containing all the intermediate points.
   * \param[in] _kmax The maximum curvature of the system.
   */
  DubinsSet(Configuration2<T> start, 
            Configuration2<T> end,
            Tuple<Point2<T> > _points,
            double _kmax=KMAX) 
  {
    this->Kmax=_kmax;
    Tuple<Angle> angles;

    #ifdef DEBUG
      cout << "Considered points: " << endl;
      cout << _points << endl;
      cout << endl;
    #endif

    // vector<Point2<T> >new_points=reduce_points(_points);
    // _points.eraseAll();
    // for (auto p : new_points){
    //   _points.add(p);
    // }

    // #ifdef DEBUG
    //   cout << "Considered points: " << endl;
    //   cout << _points << endl;
    //   cout << endl;
    // #endif

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
      find_best(_points, angles, area, 8.0, _kmax);
      area=area/8.0;
      i++;
    }

    #ifdef DEBUG 
      Mat best_img(D_DIMY, D_DIMX, CV_8UC3, Scalar(255, 255, 255));
      for (auto point : _points){
        rectangle(best_img, Point(point.x()-D_INC/2+D_SHIFT, point.y()-D_INC/2+D_SHIFT), Point(point.x()+D_INC/2+D_SHIFT, point.y()+D_INC/2+D_SHIFT), Scalar(0,0,0) , -1);
      }
      for (auto dub : this->dubinses){
        dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img, D_SHIFT);
      }
      my_imshow("best", best_img, true);
      mywaitkey();
      cout << *this << endl;
    #endif
  }

  /*!
   * \brief Constructor that computes a series of `Dubins` given only `Point2` points via brute force.
   * Since this approach is based on a brute force algorithm, it's best not to use this on too many points.
   * \param[in] _points A `Tuple` containing all points.
   * \param[in] _kmax The maximum curvature to be used for all `Dubins`.
   */
  DubinsSet(Tuple<Point2<T> > _points,
            double _kmax=KMAX){
    find_best(_points, Tuple<Angle>(), 2*M_PI, 4, _kmax);
  }

  /*!
   * Function to compute the best path. This function calls `disp()` in order to calculate all possible angles, and then creates a `Dubins` for each possibility choosing the one with the minimum length.
   * \param[in] _points A `Tuple` of `Point2` through which the path should flow.
   * \param[in] _angles A `Tuple` of `Angle` containing all base `Angle`.
   * \param[in] area This is the angle around each angle to be "scanned".
   * \param[in] tries The number of discretizations that should be made.
   * \param[in] _kmax The maximum curvature to be used.
   */
  void find_best( Tuple<Point2<T> > _points,
                  Tuple<Angle>& _angles,
                  Angle area=A_2PI,
                  double tries=4.0,
                  double _kmax=KMAX)
  {
 
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
    Tuple<Tuple<Angle> > angles;

    //Create all angles to check
    COUT(_points.size())
    disp(angles, _angles, tries, inc, 1, _points.size()-2); //startPos=1 and endPos=size()-2 since I have to check for all angles except the first and the last.

    #ifdef DEBUG
      cout << "Considered angles: " << endl;
      for (auto tupla : angles){
        cout << "<";
        for (int i=0; i<tupla.size(); i++){
          cout << tupla.get(i).to_string(Angle::DEG).str() << (i==tupla.size()-1 ? "" : ", ");
        }
        cout << ">" << endl;
      }
      cout << endl;

      Mat image(D_DIMY, D_DIMX, CV_8UC3, Scalar(255, 255, 255));

      for (auto point : _points){
          rectangle(image, Point(point.x()-D_INC/2+D_SHIFT, point.y()-D_INC/2+D_SHIFT), Point(point.x()+D_INC/2+D_SHIFT, point.y()+D_INC/2+D_SHIFT), Scalar(0,0,0) , -1);
      }

      my_imshow("dubin", image, true);
      mywaitkey();
    #endif

    //Compute Dubins
    // Tuple<Tuple<Dubins<T> > > allDubins;
    this->L=DInf;
    int id=-1;
    for (int i=0; i<angles.size(); i++){
      Tuple<Dubins<T> > app;
      double l=0.0;
      Tuple<Angle> angleT=angles.get(i);
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
        id=i;
      }

      // allDubins.add(app);
    }

    #ifdef DEBUG 
      Mat best_img(D_DIMY, D_DIMX, CV_8UC3, Scalar(255, 255, 255));
      for (auto point : _points){
        rectangle(best_img, Point(point.x()-D_INC/2+D_SHIFT, point.y()-D_INC/2+D_SHIFT), Point(point.x()+D_INC/2+D_SHIFT, point.y()+D_INC/2+D_SHIFT), Scalar(0,0,0) , -1);
      }
      for (auto dub : this->dubinses){
        dub.draw(1500, 1000, 1, Scalar(255, 0, 0), best_img, D_SHIFT);
      }
      // my_imshow("best", best_img, true);
      // mywaitkey();
      cout << *this << endl;
    #endif

    if (id<0){
      cerr << "No DubinsSet coudl be computed for given points." << endl;
    }
    else {
      _angles=angles.get(id);
    }
  }

  double getLength()              { return this->L; }               ///< Returns the Length of the set of `Dubins`.
  double getKmax()                { return this->Kmax; }            ///< Returns the maximum curvature.
  double getSize()                { return this->dubinses.size(); } ///< Returns the number of `Dubins` stored.
  Tuple<Dubins<T> > getDubinses() { return this->dubinses; }        ///< Returns a `Tuple` containing all the `Dubins`.
  Configuration2<T> getBegin()    { return this->dubinses.front().begin(); }
  Configuration2<T> getEnd()      { return this->dubinses.back().end(); }

  /*!
   * This functions returns a specific `Dubins` from the set.
   * \param[in] id The position of the `Dubins` in the set.
   * \return The id-th `Dubins`.
   */
  Dubins<T> getDubins(int id){
    if (id<(this->getSize()) && id>=0){
      return dubinses.get(id);
    }
    return Dubins<T>();
  }

  Tuple<Dubins<T> > getDubinsFrom(int id){
    return ((this->getDubinses()).get(id, this->getSize()));
  }

  /*!
   * This functions returns a specific `Dubins` from the set.
   * \param[in] id The position of the `Dubins` in the set.
   * \return The id-th `Dubins`.
   */
  Dubins<T>* getDubinsPtr(int id){
    if (id<(this->getSize()) && id>=0){
      return &(dubinses[id]);
    }
    return nullptr;
  }

  void clean (){
    this->dubinses.eraseAll();
    this->Kmax=0;
    this->L=DInf;
  }

  int is_on_dubinsSet(Configuration2<T> C){
    int ret=0;
    bool ok=false;
    for (ret=0; ret<this->getSize() && !ok; ret++){
      ok=this->getDubins(ret).is_on_dubins(C);
    }
    if (!ok) ret=-1;
    return --ret;
  }

  bool addDubins(Dubins<T>* D){
    if (D->length()!=DInf){
      if (this->getSize()==0){
        this->dubinses.add(*D);
        this->L=D->length();
        this->Kmax=D->getKmax();
      }
      else {
        if (this->getKmax()!=D->getKmax()) {
          cerr << "Cannot add a Dubins with different curvature." << endl;
          return false;
        }
        else {
          if (this->getEnd()!=D->begin()){
            int pos=is_on_dubinsSet(D->begin());
            if (pos>-1){ //Check if start is inside DubinSet somewhere
              if (pos==this->getSize()-1){ //Then I remove the last Dubins, recompute it, readd it and add the new one.
                Dubins<T> app (this->getDubins(pos).begin(), D->begin(), this->getKmax());
                cout << app << endl;
                this->removeDubins(pos);
                this->dubinses.add(app);
                this->L+=app.length();
                this->dubinses.add(*D);
                this->L+=D->length();
              }
              else{
                cerr << "Cannot add a Dubins that's disconnected from the set, wrong position." << this->getEnd() << " " << D->begin() << endl;
                return false;    
              }
            }
            else {
              cerr << "Cannot add a Dubins that's disconnected from the set." << this->getEnd() << " " << D->begin() << endl;
              return false;
            }
          }
          else {
            this->dubinses.add(*D);
            this->L+=D->length();
          }
        }
      }
    }
    return true;
  }

  bool removeDubins(uint pos){
    if (pos>=this->getSize()){
      return false;
    } 
    else {
      this->L-=this->getDubins(pos).length();
      this->dubinses.remove(pos);
      return true;
    }
  }

  DubinsSet<T> copy (DubinsSet<T>* DS)
  {
    this->dubinses.eraseAll();
    this->dubinses=DS->getDubinses();
    this->Kmax=DS->getKmax();
    this->L=DS->getLength();
    return *this;
  }

  DubinsSet<T> operator= (DubinsSet<T>* DS) { return this->copy(DS); }

  DubinsSet<T> join (DubinsSet<T>* DS, 
                     int startPos = -1)
  {
    if (DS->getSize()>0){
      if (this->getKmax()!=DS->getKmax()){
        cerr << "Cannot join to DubinsSet with different curvature." << endl;
        return DubinsSet<T>();
      }
      else {
        if (startPos==-1 || startPos>=this->getSize()){ //Then I need to add all the Dubins 
          for (uint i=0; i<DS->getSize(); i++){  
            if (!(this->addDubins( DS->getDubinsPtr(i)) )) { //If an error occured then I return an empty set. 
              return DubinsSet<T>();
            }
          }
        }
        else { //Otherwise I need to split the set and then add the Dubins.
          DubinsSet<T> new_DS;
          for (uint i=0; i<startPos; i++){ 
            if (!(new_DS.addDubins(this->getDubinsPtr(i)))){
              return DubinsSet<T>();
            }
          }
          for (uint i=0; i<DS->getSize(); i++){
            if (!(new_DS.addDubins(DS->getDubinsPtr(i)))){
              return DubinsSet<T>();
            }
          }
          for (uint i=startPos; i<this->getSize(); i++){
            if (!(new_DS.addDubins(this->getDubinsPtr(i)))){
              return DubinsSet<T>();
            }
          }
          *this=new_DS;
        }
      }
    }
    return *this;
  }

  Tuple<Tuple<Tuple<Configuration2<T> > > > splitIt (double _length=PIECE_LENGTH){
    Tuple<Tuple<Tuple<Configuration2<T> > > > ret;
    for (Dubins<T> dub: this->dubinses){
      ret.add(dub.splitIt(_length, 0));
    }
    return ret;
  }

  /*! This function create a strinstream object containing infos about the `DubinsSet`.
    \returns A string stream.
  */
  stringstream to_string() {
    stringstream out;
    out << "Total length: " << L << "\n";
    for (auto dub : dubinses){
      out << dub << endl;
    }
    return out;
  }

  /*!
   * This function overload the << operator so to print with `std::cout` the values of the `DubinsSet`, that is printing all the `Dubins` stored.
   * \param[in] out The out stream.
   * \param[in] data The `DubinsSet` to print.
   * \returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, DubinsSet& data) {
    out << data.to_string().str();
    return out;
  }


};

#endif
