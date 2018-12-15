#ifndef maths_hh
#define maths_hh

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdarg> //#include <stdarg.h>
#include <sstream>
#include <string>

using namespace std;

#include<limits>
#define DInf numeric_limits<double>::infinity()
#define Epsi numeric_limits<double>::epsilon()
/*!\brief Function to compare two dubles as \f$\vert A-B\vert < \varepsilon\f$.
   \param [in] A First number.
   \param [in] B Second number.
   \param [in] E \f$\varepsilon\f$, set at `std::numeric_limits<double>::epsilon()` as default.
   \returns `true` if \f$\vert A-B\vert < \varepsilon\f$, `false` otherwise.
*/
inline bool equal (const double& A, const double& B, const double E=Epsi) {
  return std::fabs(A-B)<E;
}

/*| \brief Simple function that takes an input \f$x\f$ and returns \f$x^2\f$.
		\tparam T The type of the input and the output;
		\param[in] x The value \f$x\f$
		\returns The squared value of \f$x\f$, \f$x^2\f$
*/
template <class T>
inline T pow2 (const T x){
  return x*x;
}

#define DEGTORAD M_PI/180
#define RADTODEG 180/M_PI


/*! \brief This class allows to save and handle angles. It supports DEG and RAD, 
		operations such as addition and subtraction with operators overloading, 
		conversion from RAD to DEG and viceversa and normalization of the angle. 
*/
class Angle
{
public:
  enum ANGLE_TYPE {DEG, RAD, INVALID}; ///<The type of angle the class can handle. DEG and RAD are self-expalantory, while INVALID is used as a flag in case of problems.

private:
  ANGLE_TYPE type;
  double th;
  
public:
  Angle () : type(RAD), th(0){} ///<A void constructor to create an angle.
  /*! \brief This constructor takes the angle value and the type of angle and stores them.
  		It also normalize the angle in case is above 2pi (360°) or below 0.
  		\param[in] _th The dimension of the angle.
  		\param[in] _type The type of the angle.
  */
  Angle(double _th, ANGLE_TYPE _type) : type(_type), th(_th){ 
    normalize();
  }
  
  // ~Angle();
  
  double get () const { return th; } ///<Returns the dimension of the angle.
  ANGLE_TYPE getType () const { return type; }///<Returns the type of the angle.
  string getTypeName () const { ///<Returns a string that tells the type of angle.
  	string ret="";
  	switch(type){
  		case DEG:
  			ret="DEG"; break;
  		case RAD:
  			ret="RAD"; break;
  		default:
  			ret="INVALID";
  	}
  	return ret; 
  } 
  
  /*! \brief Set the value of the angle.
  		\tparam T The programming type for the value to be stored. It's then cast to `double`. 
  		\param[in] _th The dimension of the angle to be stored.
  */
  template <class T>
  void set (const T _th) { th=(double)_th; normalize(); }
  /*! \brief Set the type of the angle.
  		\param[in] _th The type of the angle to be stored.
  */
  void setType (ANGLE_TYPE _type) { type=_type; normalize(); } 
  
  /*! \brief Convert and store the angle from DEG to RAD.
  		\returns The value of the angle.
  */
  double degToRad (){
    if (type == DEG){
      type=RAD;
      th=(double)(th*DEGTORAD);
    }
    return th;
  }
  
  /*! \brief Converts and stores the angle from RAD to DEG.
  		\returns The value of the angle.
  */
  double radToDeg (){
    if (type == RAD){
      type=DEG;
      th=(double)(th*RADTODEG);
    }
    return th;
  }

  /*!	\brief Converts but does not store the value of the angle from DEG to RAD.
  		\returns The value of the angle
  */  
  inline double toRad () const {
    return type==DEG ? (double)(th*DEGTORAD) : th;
  }

  /*!	\brief Converts but does not store the value of the angle from RAD to DEG.
  		\returns The value of the angle
  */
  inline double toDeg () const {
    return type==RAD ? (double)(th*RADTODEG) : th;
  }
  
  static inline bool checkValue (const double th) {
    return !isnan(th) && isfinite(th);
  }

  /*!	\brief Normalize the angle, that is to set it in \f$[0, 2\pi)\f$ or \f$[0, 360°)\f$. 
      Moreover it check if the value is infinite or NaN. In this case the `type` is set to `INVALID`.
  */
  void normalize(){
    if (checkValue(th)){
      double div=0.0;
      switch (type){
        case RAD:{
          div=(double)2*M_PI;
          break;
        }
        case DEG:{
          div=(double)360.0;
          break;
        }
        default:{
          th=0;
        }
      }
      while(th<0.0){
        th+=div;
      }
      while (th>=div){
        th-=div;
      }
    }
    else {
      th=0;
      type=INVALID;
    }
  }
  
  /*! \brief Sums and angle to this one. In the process a new angle is created so `normalize()` is also called.
  		\param[in] phi The angle to be summed.
  		\returns The angle summed. 
  */
  Angle add (const Angle phi){
    double dth = th+phi.get();
    if (type!=phi.getType()){
    	dth = th+(type==RAD ? phi.toRad() : phi.toDeg()); 
    }
    Angle alpha (dth, type);
    return alpha;
  }
  /*! \brief Subtracts and angle to this one. In the process a new angle is created so `normalize()` is also called.
  		\param[in] phi The angle to be subtracted.
  		\returns The angle subtracted. 
  */
  Angle sub (const Angle phi){
  	double dth = th-phi.get();
    if (type!=phi.getType()){
    	dth = th-(type==RAD ? phi.toRad() : phi.toDeg()); 
    }
    Angle alpha (dth, type);
    return alpha;
  }

  /*! \brief Multiply and angle by a costant. In the process a new angle is created so `normalize()` is also called.
      \tparam The type of the coefficient.
      \param[in] phi The costant to use to multiply.
      \returns The angle multiplied. 
  */
  template <class T1>
  Angle mul (const T1 A){
    double dth = th*(double)A;
    Angle alpha (dth, type);
    return alpha;
  }
  /*! \brief Divide and angle by a costant. In the process a new angle is created so `normalize()` is also called.
      \tparam The type of the dividend.
      \param[in] A The costant to use to divide.
      \returns The angle divided. 
  */
  template <class T1>
  Angle div (const T1 A){
    double dth = th/(double)A;
    Angle alpha (dth, type);
    return alpha;
  }
  /*! \brief Copies an angle to this one. In the process a new angle is created so `normalize()` is also called.
  		\param[in] A The angle to be copied.
  		\returns The new angle. 
  */  
  Angle copy (const Angle phi){
    th=phi.get();
    type=phi.getType();
    return *this;
  }
  /*! This function takes the value in radiants of two angles, an using the equal function 
      for `double` calculare if they are equal or not.
      \param[in] th0 The first angle.
      \param[in] th1 The second angle.
      \returns `true` if the two angle are equal, `false` otherwise. 
  */
  bool equal (const Angle& th0, const Angle& th1){
    return ::equal(th0.toRad(), th1.toRad()) ? true : false; 
  }
  
  /*! This function overload the operator +. It simply calls the `add()` function.
  		\param[in] phi The angle to be summed.
  		\returns The angle summed. 
  */
  Angle operator+ (const Angle phi){
    return add(phi);
  }
  /*! This function overload the operator -. It simply calls the `sub()` function.
  		\param[in] phi The angle to be subtracted.
  		\returns The angle subtracted. 
  */
  Angle operator- (const Angle phi){
    return sub(phi);
  }

  /*! This function overload the operator *. It simply calls the `mul()` function.
      \tparam The type of the coefficient.
      \param[in] A The coefficient.
      \returns The angle multiplied. 
  */
  template <class T1> 
  Angle operator* (const T1 A){
    return mul(A);
  }

  /*! This function overload the operator /. It simply calls the `div()` function.
      \tparam The type of the dividend.
      \param[in] A The dividend.
      \returns The angle divided. 
  */
  template <class T1> 
  Angle operator/ (const T1 A){
    return div(A);
  }

  /*! This function overload the operator -. It simply calls the `sub()` function.
			\param[in] phi The angle to be copied.
  		\returns The new angle. 
  */  
  Angle operator= (const Angle phi){
    return copy(phi);
  }
  
  /*! This function overload the operator +=. It simply calls the `add()` function and then assign the result to this.
  		\param[in] phi The angle to be summed.
  		\returns `this`. 
  */
  Angle& operator+= (const Angle phi){
    Angle alpha=(*this).add(phi);
    (*this)=alpha;
    return (*this);
  }
  /*! This function overload the operator -=. It simply calls the `sub()` function and then assign the result to this.
  		\param[in] phi The angle to be subtracted.
  		\returns `this`. 
  */
  Angle& operator-= (const Angle phi){
    Angle alpha=(*this).sub(phi);
    (*this)=alpha;
    return (*this);
  }
  /*! This function overload the operator *=. It simply calls the `mul()` function and then assign the result to this.
      \param[in] A The coefficient.
      \returns `this`. 
  */
  template <class T>
  Angle& operator*= (const T A){
    Angle alpha=(*this).mul(A);
    (*this)=alpha;
    return (*this);
  }
  /*! This function overload the operator /=. It simply calls the `div()` function and then assign the result to this.
      \param[in] A The dividend.
      \returns `this`. 
  */
  template <class T>
  Angle& operator/= (const T A){
    Angle alpha=(*this).div(A);
    (*this)=alpha;
    return (*this);
  }
  /*! This function overload the operator ==. It simply calls the `equal()` function.
      \param[in] phi The second angle.
      \returns `true` if the two angle are equal, `false` otherwise.  
  */
  bool operator== (const Angle& phi){
    return equal(*this, phi);
  }
  /*! This function overload the operator ==. It simply calls the `equal()` function and negates it.
      \param[in] phi The second angle.
      \returns `false` if the two angle are equal, `true` otherwise.  
  */
  bool operator!= (const Angle& phi){
    return !equal(*this, phi);
  }

  /*! \brief Compute the cosine of the angle.
      \retunrs A `double` that is the cosine of the angle.
  */
  double cos() const {
    if (type==RAD){
      return ::cos(th);
    }
    else if (type==DEG){
      return ::cos(toRad());
    }
    else {
      cout << "Invalid angle" << endl;
      return 0.0;
    }
  }
  /*! \brief Compute the sine of the angle.
      \retunrs A `double` that is the sine of the angle.
  */
  double sin() const {
    if (type==RAD){
      return ::sin(th);
    }
    else if (type==DEG){
      return ::sin(toRad());
    }
    else {
      cout << "Invalid angle" << endl;
      return 0.0;
    }
  }
  /*! \brief Compute the tangent of the angle.
      \retunrs A `double` that is the tangent of the angle.
  */
  double tan() const {
    if (type==RAD){
      return ::tan(th);
    }
    else if (type==DEG){
      return ::tan(toRad());
    }
    else {
      cout << "Invalid angle" << endl;
      return 0.0;
    }
  }

  /*! \brief Cast to int
      \returns The value in RAD of the angle casted to int
  */
  operator int()    const { return (int)    toRad(); }
  /*! \brief Cast to double
      \returns The value in RAD of the angle casted to double
  */
  operator double() const { return (double) toRad(); }
  /*! \brief Cast to float
      \returns The value in RAD of the angle casted to float
  */
  operator float()  const { return (float)  toRad(); }
  /*! \brief Cast to long
      \returns The value in RAD of the angle casted to long
  */
  operator long()   const { return (long)   toRad(); }
  
  stringstream to_string () const {
    stringstream out;
    switch (getType()){
      case DEG:{
        out << get() << "°";
        break;
      }
      case RAD:{
        double phi=get()/M_PI;
        out << phi << "pi";
        break;
      }
      default:{
        out << "Something's wrong with the angle";
        break;
      }
    }
    return out;
  }

  /*! This function overload the << operator so to print with `std::cout` the most essential info, that is the dimension and the type of angle.
  		\param[in] out The out stream.
  		\param[in] data The angle to print.
  		\returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const Angle& data) {
    out << data.to_string().str();
    return out;
  }
};

extern const Angle A_2PI = Angle(6.283185, Angle::RAD);
extern const Angle A_360 = Angle(360.0-Epsi, Angle::DEG);
extern const Angle A_PI = Angle(M_PI, Angle::RAD);
extern const Angle A_180 = Angle(180, Angle::DEG);

enum DISTANCE_TYPE {EUCLIDEAN, MANHATTAN}; ///<The possible type of distance to be computed.

/*! \bried This class allows the definition and storage of tuples of different dimensions. 
		Functions to compute distance between tuples are also available.
		\tparam T The type of elements to be stored.
*/
template <class T>
class Tuple {
private:
  int n;
  vector<T> elements;
  
public:
	/*! \brief Defualt constructor.
	*/
  Tuple () : n(0) {elements.clear();}
  /*! \brief Constructors that takes the number of objectes to be stored, 
  		the objects and then stores them. 

  		\param[in] _n Number of obejctes to store.
  		\param[in] ... Objects to store.
	*/
  Tuple <T> (int _n, ...){
    n=_n;
    va_list ap;
    va_start(ap, _n);
    for (int i=0; i<n; i++){
      T temp=(T)0.0;
      if (is_same<T, float>::value){
        temp=va_arg(ap, double);
      }
      else {
        temp=va_arg(ap, T);
      }
      elements.push_back(temp);
    }
  }
  
  // ~Tuple () {elements.clear();}
  
  int size() const { return n; } ///<\returns The number of stored elements.

	/*! \brief Gets the n-th element.
			\param[in] _n The position of the element to retrieve.
			\returns The element in the n-th position or -1 if _n is greater then n or less than 0.
	*/  
  T get (const int _n) const {
    return ((_n>=0&&_n<n) ? elements.at(_n) : -1);
  }
  
  /*! \brief Adds a value at the end of the list.
  		\param[in] _new The new value to be added.
	*/
  void add (const T _new){
  	n++;
  	elements.push_back(_new);
  }

  /*! \brief Removes a value from the list.
  		\param[in] pos The position of the value to be removed.
      \returns 1 if verything went fine, 0 otherwise.
	*/
  int remove (const T pos) {
  	int res=0;
  	if (pos>=0 && pos<n){
  		res=1;
  		elements.erase(elements.begin()+pos);
  		n--;
  	}
  	return res;
  }

  /*! \brief Set a value in a certain position, or adds the element if the 
  		position equals the number of elements.
  		\param[in] pos Must be in \f$[0, n-1] \f$. If pos\f$=n\f$ then the 
  							element is added at the end of the vector.
  		\param[in] _new The new element to be set.
  		\returns 1 if everything went right, 0 if the position was greater 
  						than \f$n\f$ or less the 0.
	*/
  int set (const int pos, const T _new){
    int res=0;
    if ( pos<n && pos>=0 ) {
      elements.at(pos)=_new;
      res=1;
    }
    else if (pos==n){
    	add(_new);
    	res=1;
    }
    return res;
  }
  
  /*! \brief Function that compute the Euclidean Distance between two tuples. 
  		They must have the same number of elements.
  		\tparan T1 The type of the elements in the second Tuple.
  		\param[in] B the second Tuple to use for computing the distance.
  		\returns The Euclidean distance between the two Tuple.
	*/
  template<class T1>
  double EuDistance(const Tuple<T1> B){
    double res=-1.0;
    if (n == B.size()){
      double rad=0.0;
      for (int i=0; i<n; i++){
        rad+=pow2(elements.at(i) - B.get(i));
      }
      res=sqrt(rad);
    }
    return res;
  }

  /*! \brief Function that compute the Manhattan Distance between two tuples. 
  		They must have the same number of elements.
  		\tparan T1 The type of the elements in the second Tuple.
  		\param[in] B the second Tuple to use for computing the distance.
  		\returns The Manhattan distance between the two Tuple.
	*/
  template<class T1>
  double MaDistance(const Tuple<T1> B){
    double res=-1.0;
    if (n == B.size()){
      double sum=0.0;
      for (int i=0; i<n; i++){
        sum+=fabs(get(i)-B.get(i));
      }
      res=sum;
    }
    return res;
  }
  
  /*!	\brief Wrapper to compute different distances. 
		  They must have the same number of elements.
  		\tparan T1 The type of the elements in the second Tuple.
  		\param[in] B The second Tuple to use for computing the distance.
  		\param[in] dist The type of distance to be computed.
  		\returns The distance between the two Tuple.
  */
  template<class T1>
  double distance (const Tuple<T1> B, const DISTANCE_TYPE dist=EUCLIDEAN){
    double ret=0.0;
    switch (dist){
      case (EUCLIDEAN): {
        ret=EuDistance(B);
        break;
      }
      case (MANHATTAN): {
        ret=MaDistance(B);
        break;
      }
      default: {
        ret=-1.0;
      }
    }
    return ret;
  }
  
  stringstream to_string() const {
    stringstream out;
    out << '<';
    for (int i=0; i<size(); i++){
      out << get(i) << ((i!=size()-1) ? ", " : "");
    }
    out << ">";
    
    return out;
  }

  /*!	\brief Overload of operator << to output the content of the tuple.
			\param[in] out The output stream.
  		\param[in] data The Tuple to print.
  		\returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const Tuple<T>& data) {
    out << data.to_string().str();
    return out;
  }
};


/*!	\brief Class that stores two value to construct a point in 2D. The value is saved in a Tuple.
		\tparam T The type of the coordinates to be stored.
*/
template <class T>
class Point2 //: public Tuple<T>
{
private:
  Tuple<T> values;
  
public:
  Point2() {values=Tuple<T>(2, 0, 0);} ///<Default constructor to build an empty Tuple.
  /*!	\brief Constructor that taked to elements and builds a point.
  		\param[in] _x The abscissa coordinate.
  		\param[in] _y The ordinate coordinate.
  */
  Point2(const T _x, const T _y) {
    values=Tuple<T> (2, _x, _y);
  }
  
  T x() const {return values.get(0);} ///< \returns The abscissa coordinate
  T y() const {return values.get(1);} ///< \returns The ordinate coordinate
  
  /*! \brief Set the abscissa value.
  		\param[in] _x The new abscissa value
  		\returns 1 if it was successful, 0 otherwise.
  */
  int x(const T _x) {return values.set(0, _x);}
  /*! \brief Set the ordinate value.
  		\param[in] _x The new ordinate value
  		\returns 1 if it was successful, 0 otherwise.
  */
  int y(const T _y) {return values.set(1, _y);}
  
  /*! \brief This function compute the offset of the point given a vector, 
  		that is the lenght of the vector and its angle. The angle must be an 
  		`Angle` variable.
			\tparam[T1] The type of the lenght of the vector.
			\param[in] _offset The lenght of the vector.
			\param[in] th The angle of the vector.
			\returns 1 if everything went fine, 0 otherwise.
  */
  template <class T1>
  int offset(const T1 _offset, const Angle th){
    double dth = th.getType()==Angle::RAD ? th.get() : th.toRad();
    double dx=_offset*cos(dth);
    double dy=_offset*sin(dth);
    if (is_same<T, int>::value){ //Since casting truncates the value, adding 0.5 is the best way to round the numbr
      dx+=0.5;
      dy+=0.5;
    }
    T _x=x()+(T)dx;
    T _y=y()+(T)dy;
    return (values.set(0, _x) &&
            values.set(1, _y));
  }
  
  /*! \brief This function compute an offset given another point made 
  						of the abscissa offset and the ordinate offset. 
  		\param[in] p The point with the offsets.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  int offset (const Point2<T> p){
    return (values.set(0, p.x()+x()) &&
            values.set(1, p.y()+y())); 
  }
  
  /*! \brief This function compute an offset given a `Tuple` made 
  						of the abscissa offset and the ordinate offset. 
  		\param[in] p The `Tuple` with the offsets. Its dimension must be 2.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  int offset (const Tuple<T> p){
    int res=0;
    if (p.size()==2){
      res= (values.set(0, p.get(0)+x())) &&
      			values.set(1, p.get(1)+y());
    }
    return res;
  }
  
  /*! \brief This function compute an offset for the abscissa.
  		\param[in] _offset The offset.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  int offset_x(const T _offset){
    return values.set(0, _offset+x());
    // return values.set(0, _offset+values.get(0));
  }
  
  /*! \brief This function compute an offset for the ordinate.
  		\param[in] _offset The offset.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  int offset_y(const T _offset){
    return values.set(1, _offset+y());
    // return values.set(1, _offset+values.get(1));
  }
  
  /*!	\brief Wrapper to compute different distances. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B The second `Point2` to use for computing the distance.
  		\param[in] dist The type of distance to be computed.
  		\returns The distance between the two points.
  */
  template<class T1>
  double distance (Point2<T1> B, DISTANCE_TYPE dist=EUCLIDEAN){
    return values.distance(Tuple<T1>(2, B.x(), B.y()), dist);
  }

  /*! \brief Function that compute the Manhattan Distance between two points. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B the second `Point2` to use for computing the distance.
  		\returns The Manhattan distance between the two points.
	*/
  template<class T1>
  double MaDistance (Point2<T1> B){
    return values.MaDistance(Tuple<T1>(2, B.x(), B.y()));
  }
  
  /*! \brief Function that compute the Euclidean Distance between two points. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B the second `Point2` to use for computing the distance.
  		\returns The Euclidean distance between the two points.
	*/
  template<class T1>
  double EuDistance (Point2<T1> B){
    return values.EuDistance(Tuple<T1>(2, B.x(), B.y()));
  }
  
  stringstream to_string () const {
    stringstream out;
    out << "x: " << x();
    out << ", y: " << y();
    return out;
  }

  /*!	\brief Overload of operator << to output the content of a `Point2`.
			\param[in] out The output stream.
  		\param[in] data The `Point2` to print.
  		\returns An output stream to be printed.
  */
  friend ostream& operator<< (ostream& out, const Point2<T> &data){
    out << data.to_string().str();
    return out;
  }

  // ~Point2(){delete values;}
};


/*!	\brief This class stores a configuration, that is a point and an angle.
		\tparam T1 The type of the coordinates.
*/
template <class T1>
class Configuration2 //: public Tuple<T>
{
private:
  Point2<T1> coord;
  Angle th;
  
public:
	/*!	\brief Default constructor that use as point (0,0) and as angle 0 RAD. 
	*/
  Configuration2() {
    coord=Point2<T1>();
    th=Angle(0.0, Angle::RAD);
  }
  /*!	\brief Default constructor that takes the coordinates, the angle, and
  					stores them.
  		\param[in] _x The abscissa coordinate.
  		\param[in] _y The ordinate coordinate.
  		\param[in] _th The angle.
	*/
  Configuration2(	const T1 _x, 
  								const T1 _y, 
  								const Angle _th) {
    coord=Point2<T1>(_x, _y);
    th=_th;
  }
	/*!	\brief Default constructor that takes the point, the angle, and
  					stores them.
  		\param[in] P The coordinates.
  		\param[in] _th The angle.
	*/
  Configuration2(	const Point2<T1> P, 
  								const Angle _th) {
    coord=Point2<T1>(P.x(), P.y());
    th=_th;
  }
  
  T1 x() const {return coord.x();} ///<\returns The abscissa coordinate.
  T1 y() const {return coord.y();} ///<\returns The ordinate coordinate.
  Angle angle() const {return th;} ///<\returns The angle.
  
  /*! \brief This function stores a new value for the abscissa. 
  		\param[in] _x The value to be stored.
  		\returns 1 if everything went ok, 0 otherwise.
  */
  int x(const T1 _x)   {return coord.x(_x);}
  
  /*! \brief This function stores a new value for the ordinate. 
  		\param[in] _y The value to be stored.
  		\returns 1 if everything went ok, 0 otherwise.
  */  
  int y(const T1 _y)   {return coord.y(_y);}

  /*! \brief This function stores a new value for the angle. 
  		\param[in] _th The value to be stored.
  		\returns 1 if everything went ok, 0 otherwise.
  */
  void angle(const Angle _th)   {th=Angle(_th.get(), _th.getType());}
  
  /*! \brief This function compute the offset of the point given a vector, 
  		that is the lenght of the vector and its angle. The angle must be an 
  		`Angle` variable. It takes also another `Angle` to change the `Angle`
  		in the configuration. 
			\tparam[T2] The type of the lenght of the vector.
			\param[in] _offset The lenght of the vector.
			\param[in] phi The angle of the vector.
			\param[in] _th The offset for the `Angle` in the configuration.
			\returns 1 if everything went fine, 0 otherwise.
  */
  template <class T2>
  int offset(	const T2 _offset, 
  						const Angle phi, 
  						const Angle _th){
    th+=_th;
    return coord.offset(_offset, phi);
  }
  /*! \brief This function compute the offset of the point given another
  		`Configuration2`.
			\param[in] p The configuration containing the offsets.
			\returns 1 if everything went fine, 0 otherwise.
  */
  int offset (Configuration2<T1> p){
    th+=p.angle();
    return coord.offset(Point2<T1>(p.x(), p.y()));
  }
  
  /*! \brief This function compute the offset of the point given a `Point2` 
  		containing the offsets for the abscissa and the ordiante and an `Angle` 
  		to change the `Angle` in the configuration. 
			\param[in] p The point containing the offsets.
			\param[in] _th The offset for the `Angle` in the configuration. 
					It's set to 0 as default so to easily change just the coordinates.
			\returns 1 if everything went fine, 0 otherwise.
  */
  int offset (Point2<T1> p, 
  						const Angle _th=Angle()){
    th+=_th;
    return coord.offset(p);
  }
  
  /*!	\brief Function to add an offset to the abscissa.
  		\param[in] _offset The offset.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  int offset_x(const T1 _offset){
    return coord.offset_x(_offset);
    // return values.set(0, _offset+values.get(0));
  }
  
  /*!	\brief Function to add an offset to the ordinate.
		\param[in] _offset The offset.
		\returns 1 if everything went fine, 0 otherwise.
	*/
  int offset_y(const Angle _offset){
    return coord.offset_y(_offset);
    // return values.set(1, _offset+values.get(1));
  }
  
  /*!	\brief Function to add an offset to the angle.
		\param[in] _offset The offset.
		\returns 1 if everything went fine, 0 otherwise.
	*/
  void offset_angle(const Angle _th){
    th+=_th;
  }
  
  /*!	\brief Wrapper to compute different distances. 
  		\tparan T2 The type of the elements in the second `Configuration2`.
  		\param[in] B The second `Configuration2` to use for computing the distance.
  		\param[in] dist The type of distance to be computed.
  		\returns The distance between the two configurations.
  */
  template <class T2>
  Tuple<double> distance (Configuration2<T2> B, DISTANCE_TYPE dist_type=EUCLIDEAN){
    double dist =	coord.distance(Point2<T2>(B.x(), B.y()), dist_type);
    double dth	=	B.angle()-th;
    return Tuple<double>(2, dist, dth);
  }
  
  /*! \brief Function that compute the Euclidean Distance between two configurations. 
  		\tparan T2 The type of the elements in the second `Configuration2`.
  		\param[in] B the second `Configuration2` to use for computing the distance.
  		\returns The Euclidean distance between the two configurations.
	*/
  template <class T2>
  Tuple<double> EuDistance (Configuration2<T2> B){
    double dist =	coord.EuDistance(Point2<T2>(B.x(), B.y()));
    double dth	= B.angle()-th;
    return Tuple<double>(2, dist, dth);
  }
  
  /*! \brief Function that compute the Manhattan Distance between two configurations. 
  		\tparan T2 The type of the elements in the second `Configuration2`.
  		\param[in] B the second `Configuration2` to use for computing the distance.
  		\returns The Manhattan distance between the two configurations.
	*/
  template <class T2>
  Tuple<double> MaDistance (Configuration2<T2> B){
    double dist =	coord.MaDistance(Point2<T2>(B.x(), B.y()));
    double dth	=	B.angle()-th;
    return Tuple<double>(2, dist, dth);
  }
  
  /*!	\brief Overload of operator << to output the content of a `Configuration2`.
			\param[in] out The output stream.
  		\param[in] data The `Configuration2` to print.
  		\returns An output stream to be printed.
  */
  stringstream to_string() const {
    stringstream out;
    out << "x: " << x();
    out << ", y: " << y();
    out << ", th: " << angle();
    return out;
  }

  friend ostream& operator<< (ostream& out, const Configuration2<T1> &data){
    out << data.to_string().str();
    return out;
  }
  
  // ~Configuration2(){delete coord;}
};


#endif