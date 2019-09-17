#ifndef MATHS_HH
#define MATHS_HH

#include <utils.hh>

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdarg> //#include <stdarg.h>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

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

template<class T>
inline T my_max (const T& A, const T& B){
  return (A>B ? A : B);
}

template<class T>
inline T my_min (const T& A, const T& B){
  return (A<B ? A : B);
}

const double DEGTORAD=(M_PI/180.0);
const double RADTODEG=(180.0/M_PI);

/*! \brief This class allows to save and handle angles. It supports DEG and RAD, 
		operations such as addition and subtraction with operators overloading, 
		conversion from RAD to DEG and viceversa and normalization of the angle. 
*/
class Angle
{
public:
  enum ANGLE_TYPE {DEG, RAD, INVALID}; ///<The type of angle the class can handle. DEG and RAD are self-expalantory, while INVALID is used as a flag in case of problems.

private:
  ANGLE_TYPE type; ///<The type of the angle.
  double th; ///<The value of the angle.
  
public:
  Angle () : type(RAD), th(0){} ///<A void constructor to create an angle.
  /*! \brief This constructor takes the angle value and the type of angle and stores them.
  		It also normalize the angle in case is above 2pi (360°) or below 0.
  		\param[in] _th The dimension of the angle.
  		\param[in] _type The type of the angle.
  */

  Angle(double _th, ANGLE_TYPE _type=RAD) : type(_type), th(_th){ 
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
    return (type==DEG ? (double)(th*DEGTORAD) : th);
  }

  /*!	\brief Converts but does not store the value of the angle from RAD to DEG.
  		\returns The value of the angle
  */
  inline double toDeg () const {
    return (type==RAD ? (double)(th*RADTODEG) : th);
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
    double dth=0.0;
    switch(type){
      case RAD:{
        dth=get()+phi.toRad();
        break;
      }
      case DEG:{
        dth=get()+phi.toDeg();
        break;
      }
      default:{
        cerr << "Wrong type" << endl;
      }
    }
    Angle alpha (dth, type);
    return alpha;
  }
  /*! \brief Subtracts and angle to this one. In the process a new angle is created so `normalize()` is also called.
  		\param[in] phi The angle to be subtracted.
  		\returns The angle subtracted. 
  */
  Angle sub (const Angle phi){
  	double dth=0.0;
    switch(type){
      case RAD:{
        dth=get()-phi.toRad();
        break;
      }
      case DEG:{
        dth=get()-phi.toDeg();
        break;
      }
      default:{
        cerr << "Wrong type" << endl;
      }
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

  /*! This function overload the operator =. It simply calls the `copy()` function.
			\param[in] phi The angle to be copied.
  		\returns The new angle. 
  */  
  Angle operator= (const Angle phi){
    return this->copy(phi);
  }

  Angle operator= (const double phi){
    return Angle(phi, RAD);
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

  /*! This function takes an angle to copare, an using the `equal` function 
      for `double`s calculates if it is equal or not to `this`.
      \param[in] phi The angle to compare.
      \returns `true` if the two angle are equal, `false` otherwise. 
  */
  bool equal (const Angle& phi){
    return ::equal(this->toRad(), phi.toRad(), 0.0000001) ? true : false; 
  }

  /*! This function takes the value in radiants of an angle and compares it with this.
      \param[in] phi The angle to compare.
      \returns `true` if this is less than phi, `false` otherwise. 
  */
  bool less (const Angle& phi){
    return this->toRad()<phi.toRad(); 
  }

  /*! This function takes the value in radiants of an angle and compares it with this.
      \param[in] phi The angle to compare.
      \returns `true` if this is more than phi, `false` otherwise. 
  */
  bool greater (const Angle& phi){
    return this->toRad()>phi.toRad(); 
  }

  /*! This function overload the operator ==. It simply calls the `equal()` function.
      \param[in] phi The second angle.
      \returns `true` if the two angle are equal, `false` otherwise.  
  */
  bool operator== (const Angle& phi){
    return this->equal(phi);
  }

  /*! This function overload the operator !=. It simply calls the `equal()` function and negates it.
      \param[in] phi The second angle.
      \returns `false` if the two angle are equal, `true` otherwise.  
  */
  bool operator!= (const Angle& phi){
    return !(this->equal(phi));
  }

  /*! This function overload the operator <. It simply calls the `less()` function.
      \param[in] phi The second angle.
      \returns `true` if the first angle (this) is less than the second one, `false` otherwise.  
  */
  bool operator< (const Angle& phi){
    return this->less(phi);
  }

  /*! This function overload the operator >. It simply calls the `greater()` function.
      \param[in] phi The second angle.
      \returns `true` if the first angle (this) is greater than the second one, `false` otherwise.  
  */
  bool operator> (const Angle& phi){
    return this->greater(phi);
  }

  /*! This function overload the operator <. It simply calls the `less()` function and `equal()` function.
      \param[in] phi The second angle.
      \returns `true` if the first angle (this) is less or equal than the second one, `false` otherwise.  
  */
  bool operator<= (const Angle& phi){
    return (this->less(phi) || this->equal(phi));
  }

  /*! This function overload the operator <. It simply calls the `greater()` function and `equal()` function.
      \param[in] phi The second angle.
      \returns `true` if the first angle (this) is greater or equal than the second one, `false` otherwise.  
  */
  bool operator>= (const Angle& phi){
    return (this->greater(phi) || this->equal(phi));
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

  /*! This function create a strinstream object containing the most essential info, that is the dimension and the type of angle.
      \param[in] The type of values to be printed. Default is set to INVALID and it'll print the data of the `Angle` as it was saved.
      \returns A string stream.
  */
  stringstream to_string (ANGLE_TYPE _type=INVALID) const {
    stringstream out;
    ANGLE_TYPE search=_type==INVALID ? this->getType() : _type;
    switch (search){
      case DEG:{
        out << this->toDeg() << "°";
        break;
      }
      case RAD:{
        double phi=toRad()/M_PI;
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

#define A_2PI Angle(6.2831853071-Epsi, Angle::RAD)  ///<Default Angle for 2pi rad
#define A_360 Angle(360.0-Epsi, Angle::DEG)         ///<Default Angle for 360 degree
#define A_PI Angle(M_PI, Angle::RAD)                ///<Default Angle for pi rad
#define A_180 Angle(180, Angle::DEG)                ///<Defualt Angle for 180 degree
#define A_PI2 Angle(M_PI/2.0, Angle::RAD)           ///<Default Angle for pi/2 rad
#define A_90 Angle(90, Angle::DEG)                  ///<Defualt Angle for 90 degree
#define A_DEG_NULL Angle(0, Angle::DEG)             ///<Default Angle for 0 rad
#define A_RAD_NULL Angle(0, Angle::RAD)             ///<Defualt Angle for 0 degree

enum DISTANCE_TYPE {EUCLIDEAN, MANHATTAN}; ///<The possible type of distance to be computed.

// extern double elapsedTuple;
// extern double elapsedTupleSet;
/*! \bried This class allows the definition and storage of tuples of different dimensions. 
		Functions to compute distance between tuples are also available.
		\tparam T The type of elements to be stored.
*/
template <class T>
class Tuple {
private:
  int n; ///<The number of elements.
  vector<T> elements; ///<The elements.
  
public:
	/*! \brief Defualt constructor.
	*/
  Tuple () : n(0) {elements.clear();}
  /*! \brief Constructors that takes the number of objectes to be stored, 
  		the objects and then stores them. 
      For compatibility problem we strongly suggest to use this constructor 
      only with standard types or types that can be promotted to one of the standard ones.
      For any other type we suggest to use an empty constructor and then use 
      the `add()` function.

  		\param[in] _n Number of obejctes to store.
  		\param[in] ... Objects to store.
	*/
  Tuple <T> (int _n, ...){
    this->n=_n;
    auto start=Clock::now();
    this->elements.reserve(this->n);
    auto stop=Clock::now();
    // elapsedTupleSet+=CHRONO::getElapsed(start, stop);
    va_list ap;
    va_start(ap, _n);
    start=Clock::now();
    for (int i=0; i<this->n; i++){
      T temp;
      if (std::is_same<T, float>::value){
        temp=va_arg(ap, double);
      }
      else {
        temp=va_arg(ap, T);
      }
      this->elements.push_back(temp);
    }
    stop=Clock::now();
    // elapsedTuple+=CHRONO::getElapsed(start, stop);
  }

  /**
   * @brief      Constructor that takes a vector with elements and stores it.
   *
   * @param[in]  v     The vector to store.
   */
  Tuple<T> (std::vector<T> v){
    this->elements=v;
    this->n=v.size();
  }
  
  // ~Tuple () {elements.clear();}
  
  int size() const { return (n==(int)elements.size() ? n : -1); } ///<\returns The number of stored elements. -1 if the Tuple has a different number of elements.

	/*! \brief Gets the n-th element.
			\param[in] _n The position of the element to retrieve.
			\returns The element in the n-th position or an empty costructor if _n is greater then n or less than 0.
	*/  
  T get (const int _n) const {
    return ((_n>=0&&_n<size()) ? elements.at(_n) : T());
  }

  /**
   * @brief      Function that returns a `Tuple` with elements.
   *
   * @param[in]  start  The starting position
   * @param[in]  end    The ending position
   *
   * @return     A `Tuple` containing the element from the start-th position to the end-th.
   */
  Tuple<T> get(const uint start, const uint end){
    Tuple<T> ret;
    if (start>end){
      throw MyException<string>(GENERAL, "End is bigger than start", __LINE__, __FILE__);
    }
    else {
      for (uint i=start; i<end; i++){
        ret.add(this->get(i));
      }
    }
    return ret;
  }


  T front ()  { return this->elements.front(); } ///< \return The first element in the `Tuple`.
  T back ()   { return this->elements.back(); } ///< \return The last element in the `Tuple`.
  
  /**
   * @brief      A function that search for an element in the `Tuple` and returns it. 
   *
   * @param[in]  _el   The element you are looking for.
   *
   * @return     The position of the element. -1 if no such element was found.
   */
  int find (T _el){
    for (int i=0; i<this->size(); i++){
      if (this->get(i)==_el){
        return i;
      }
    }
    return -1;
  }

  /*! \brief Adds a value at the end of the list.
  		\param[in] _new The new value to be added.
	*/
  void add (const T _new){
  	n++;
  	elements.push_back(_new);
  }
  
  /**
   * @brief      Adds a value. but only if it is not already present.
   *
   * @param[in]  _el     The element to add. 
   * @param[in]  _throw  If an exception can be thrown.
   */
  void addIfNot(T _el, bool _throw=false){
    int id=find(_el); 
    if (id<0){
      this->add(_el);
    }
    else if (_throw){
      throw MyException<T>(EXCEPTION_TYPE::EXISTS, _el, id);
    }
  }

  /*! \brief Removes a value from the list.
  		\param[in] pos The position of the value to be removed.
      \returns `true` if verything went fine, `false` otherwise.
	*/
  bool remove (const uint pos) {
  	bool res=false;
  	if (pos>=0 && ((int)pos)<n){
  		res=true;
  		elements.erase(elements.begin()+pos);
  		n--;
  	}
  	return res;
  }

  /**
   * @brief      Removes all the element from a position onwards.
   *
   * @param[in]  pos   The position from which to remove the elements.
   *
   * @return     `true` if the elements could be removed, `false` otherwise.
   */
  bool remove_from (const uint pos){
    bool res=true;
    if (pos<this->size()){
      for (uint i=pos; i<this->size() && res; i++){
        res=this->remove(i);
      }      
    }
    else {
      res=false;
    }
    return res;
  }

  /*! \brief Removes all values from the `Tuple`.
  */
  void eraseAll (){
    this->elements.clear();
    this->n=0;
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

  /**
   * @brief      Function that adds an element at the head of the vector.
   *
   * @param[in]  _new  The element to be added.
   */
  void ahead (const T _new){
    Tuple<T> newT;
    newT.add(_new);
    for (auto el : *this){
      newT.add(el);
    }
    this->add(T());
    *this=newT;
  }
  
  /*! \brief Copy a Tuple into another one.
    \param [in] A Tuple to be coppied.
    \returns this. 
  */
  Tuple<T> copy (const Tuple<T>& A){
    this->eraseAll();
    for (int i=0; i<A.size(); i++){
      this->add(A.get(i));
    }
    return *this;
  }
  /*! \brief Overload of the = operator. Just calls `copy`.
      \param [in] A Tuple to be coppied.
      \returns this. 
  */
  Tuple<T> operator= (const Tuple<T>& A){
    return this->copy(A);
  }

  /**
   * @brief      Function that takes two `Tuple`s and verifies if they contain the same values. 
   * @param[in]  _t    The `Tuple` to compare. 
   *
   * @return     `true` if the two `Tuple`s have the same element, `false` otherwise.
   */
  bool equal(Tuple<T> _t){
    if (this->size()!=_t.size()){ return false; }

    for (int i=0; i<this->size(); i++){
      if (this->get(i) != _t.get(i)){
        return false;
      }
    }
    return true;
  }
  
  /*! This function overload the operator ==. It simply calls the `equal()` function.
    \param[in] _t The second `Tuple`.
    \returns `true` if the first `Tuple` (this) is equal to the second one, `false` otherwise.  
  */
  bool operator== (Tuple<T> _t){
    return equal(_t);
  }

  Tuple<T> sum(Tuple<T> t){
    if (this->size()!=t.size()){
      throw MyException<int>(EXCEPTION_TYPE::SIZE, this->size, t.size());
    }
    for (int i=0; i<this->size(); i++){
      this->set(i, (this->get(i)+t.get(i)));
    }
    return (*this);
  }

  /**
   * @brief      Function to sum a value to all the elements in the `Tuple`.
   *
   * @param[in]  inc   The increment
   *
   * @return     A `Tuple` (this) containing the new values.
   */
  Tuple<T> sum(T inc){
    for (int i=0; i<this->size(); i++){
      this->set(i, (this->get(i)+inc));
    }
    return (*this);
  }

  /*! This function overload the operator +. It simply calls the `sum()` function.
    \param[in] _t The increment.
    \returns A `Tuple` (this) containing the new values.  
  */
  Tuple<T> operator+ (T inc){
    return this->sum(inc);
  }

  /*! This function overload the operator +. It simply calls the `sum()` function.
    \param[in] _t The increment.
    \returns A `Tuple` (this) containing the new values.  
  */
  Tuple<T>& operator+= (T inc){
    return this->sum(inc);
  }

  /**
   * @brief      Function to multiply one by one the values from this to the values of a `Tuple`.
   *
   * @param[in]  inc   The multiplier `Tuple`
   *
   * @return     A `Tuple` (this) containing the new values.
   */
  Tuple<T> mul(Tuple<T> t){
    if (this->size()!=t.size()){
      throw MyException<int>(EXCEPTION_TYPE::SIZE, this->size, t.size());
    }
    for (int i=0; i<this->size(); i++){
      this->set(i, (this->get(i)*t.get(i)));
    }
    return (*this);
  }

  /**
   * @brief      Function to multiply a value to all the elements in the `Tuple`.
   *
   * @param[in]  inc   The multiplier
   *
   * @return     A `Tuple` (this) containing the new values.
   */
  Tuple<T> mul(T inc){
    for (int i=0; i<this->size(); i++){
      this->set(i, (this->get(i)*inc));
    }
    return (*this);
  }

  /*! This function overload the operator *. It simply calls the `mul()` function with only a multiplier and not a `Tuple`.
    \param[in] _t The increment.
    \returns A `Tuple` (this) containing the new values.  
  */
  Tuple<T> operator* (T inc){
    return this->mul(inc);
  }

  /*! This function overload the operator *=. It simply calls the `mul()` function with only a multiplier and not a `Tuple`.
    \param[in] _t The increment.
    \returns A `Tuple` (this) containing the new values.  
  */
  Tuple<T>& operator*= (T inc){
    return this->mul(inc);
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

  /*! This function create a strinstream object containing the values of the Tuple.
    \returns A string stream.
  */
  stringstream to_string(string _prefix="") const {
    stringstream out;
    string prefix=_prefix.back()=='/' && _prefix!="" ? _prefix : _prefix+"/";
    for (int i=0; i<size(); i++){
      out << _prefix << get(i) << ((i!=size()-1) ? ", " : "");
    }
    return out;
  }

  /*!	\brief Overload of operator << to output the content of the tuple.
			\param[in] out The output stream.
  		\param[in] data The Tuple to print.
  		\returns An output stream to be printed.
  */
  friend ostream& operator<<(ostream &out, const Tuple<T>& data) {
    out << '<';
    out << data.to_string().str();
    out << ">";
    return out;
  }

  /**
   * @brief      Returns a standard string of the object.
   *
   * @return     Standard string of the object.
   */
  string to_std_string() const {
    return this->to_string().str();
  }
  
  /**
   * @brief      Overload of operator std::string(). It simply calls the function `to_std_string()`.
   */
  operator std::string() const {
    return to_std_string();
  }

  /*!\brief Overload of cast to vector of same type.
   * \return A vector containing the values of elements.
   */
  operator vector<T> () const {
    return this->elements;
  }

  /*!\brief Overload of cast to vector of different type.
   * \tparam Type of vector to cast to.
   * \return A vector containing the values of elements.
   */
  template <class T1>
  operator vector<T1> () const {
      vector<T1> v;
      for (int i=0; i<this->size(); i++){
        v.push_back((T1)(this->elements[i]));
      }
      return v;
  }

  /*!\brief Overloading [] operator to access elements in array style 
   * \param[in] index Id of value to get.
   * \returns Value at id position.
   */
  T &operator[](int index) {
    if (index >= this->size()) 
    { 
      throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Array index out of bound, exiting.", __LINE__, __FILE__); 
    } 
    return this->elements[index]; 
  } 


  #define tupleIter typename vector<T>::iterator
  #define tupleConstIter const typename vector<T>::iterator

  //////FOREACH CODE///////
  tupleIter begin()           { return this->elements.begin(); } ///<Iterator.\returns the elements.begin() iterator.
  tupleConstIter begin() const{ return this->elements.begin(); } ///<Const iterator.\returns the elements.begin() iterator.

  tupleIter end()             { return this->elements.end(); } ///<Iterator.\returns the elements.end() iterator.
  tupleConstIter end() const  { return this->elements.end(); } ///<Const iterator.\returns the elements.begin() iterator.
};


/*!	\brief Class that stores two value to construct a point in 2D. The value is saved in a Tuple.
		\tparam T The type of the coordinates to be stored.
*/
template <class T>
class Point2 //: public Tuple<T>
{
private:
  // Tuple<T> values; ///<The values stored.
  T X, Y;
public:
  Point2() : X(0), Y(0) {} ///<Default constructor to build an empty Tuple.
  /*!	\brief Constructor that taked to elements and builds a point.
  		\param[in] _x The abscissa coordinate.
  		\param[in] _y The ordinate coordinate.
  */
  Point2(const T _x, const T _y) : X(_x), Y(_y){}

  /*!\brief Constructor that takes a cv::Point and returns a Point2.
    \param[in] p The cv::Point to be copied.
  */
  Point2(const cv::Point p) : X(p.x), Y(p.y){}
  
  T x() const {return X;} ///< \returns The abscissa coordinate
  T y() const {return Y;} ///< \returns The ordinate coordinate
  
  /*! \brief Set the abscissa value.
  		\param[in] _x The new abscissa value
  		\returns 1 if it was successful, 0 otherwise.
  */
  void x(const T _x) {X=_x;}
  /*! \brief Set the ordinate value.
  		\param[in] _x The new ordinate value
  		\returns 1 if it was successful, 0 otherwise.
  */
  void y(const T _y) {Y=_y;}
  
  /*! \brief This function compute the offset of the point given a vector, 
  		that is the lenght of the vector and its angle. The angle must be an 
  		`Angle` variable.
			\tparam[T1] The type of the lenght of the vector.
			\param[in] _offset The lenght of the vector.
			\param[in] th The angle of the vector.
			\returns 1 if everything went fine, 0 otherwise.
  */
  template <class T1>
  Point2<T> offset(const T1 _offset, const Angle th){
    double dx=_offset*th.cos();
    double dy=_offset*th.sin();
    if (is_same<T, int>::value){ //Since casting truncates the value, adding 0.5 is the best way to round the numbr
      dx+=0.5;
      dy+=0.5;
    }
    T _x=this->x()+(T)dx;
    T _y=this->y()+(T)dy;
    this->x(_x); this->y(_y);
    return *this;
  }
  
  /*! \brief This function compute an offset given another point made 
  						of the abscissa offset and the ordinate offset. 
  		\param[in] p The point with the offsets.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  Point2<T> offset (const Point2<T> p){
    x(p.x()+x());
    y(p.y()+y()); 
    return *this;
  }
  
  /*! \brief This function compute an offset given a `Tuple` made 
  						of the abscissa offset and the ordinate offset. 
  		\param[in] p The `Tuple` with the offsets. Its dimension must be 2.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  Point2<T> offset (const Tuple<T> p){
    int res=0;
    if (p.size()==2){
      x(p.get(0)+x());
      y(p.get(1)+y());
    }
    return *this;
  }
  
  /*! \brief This function compute an offset for the abscissa.
  		\param[in] _offset The offset.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  Point2<T> offset_x(const T _offset){
    x(_offset+x());
    // return values.set(0, _offset+values.get(0));
    return *this;
  }
  
  /*! \brief This function compute an offset for the ordinate.
  		\param[in] _offset The offset.
  		\returns 1 if everything went fine, 0 otherwise.
  */
  Point2<T> offset_y(const T _offset){
    y(_offset+y());
    // return values.set(1, _offset+values.get(1));
    return *this;
  }
  
  /*!	\brief Wrapper to compute different distances. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B The second `Point2` to use for computing the distance.
  		\param[in] dist The type of distance to be computed.
  		\returns The distance between the two points. If something went wrong the return is -1.0.
  */
  template<class T1>
  double distance (Point2<T1> B, DISTANCE_TYPE dist=EUCLIDEAN){
    switch(dist){
      case EUCLIDEAN: return EuDistance(B); break;
      case MANHATTAN: return MaDistance(B); break;
    }
    return -1.0;
  }

  /*! \brief Function that compute the Manhattan Distance between two points. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B the second `Point2` to use for computing the distance.
  		\returns The Manhattan distance between the two points.
	*/
  template<class T1>
  double MaDistance (Point2<T1> B){
    return (x()-B.x())+(y()-B.y());
  }
  
  /*! \brief Function that compute the Euclidean Distance between two points. 
  		\tparan T1 The type of the elements in the second `Point2`.
  		\param[in] B the second `Point2` to use for computing the distance.
  		\returns The Euclidean distance between the two points.
	*/
  template<class T1>
  double EuDistance (Point2<T1> B){
    return sqrt(pow2(x()-B.x())+pow2(y()-B.y()));
  }
  
  /**
   * @brief      Returns a string representation of the object.
   *
   * @return     String representation of the object.
   */
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
    out << "[" << data.to_string().str() << "]";
    return out;
  }

  /*! \brief Copy a point into another one.
    \param [in] A point to be coppied.
    \returns this. 
  */
  Point2<T> copy (const Point2<T>& A){
    x(A.x());
    y(A.y());
    return *this;
  }
  /*! \brief Overload of the = operatore. Just calls `copy`.
      \param [in] A point to be coppied.
      \returns this. 
  */
  Point2<T> operator= (const Point2<T>& A){
    return this->copy(A);
  }
  /*! \brief Equalize two points.
      \param [in] A point to be compared to.
      \returns true if the two points are equal. 
  */
  bool equal (const Point2<T>& A){
    return ::equal(this->x(), A.x()) && ::equal(this->y(), A.y()); 
  }
  /*! \brief Overload of the == operator. Just calls `equal`.
      \param [in] A point to be compared to.
      \returns true if the two configurations are equal. 
  */
  bool operator== (const Point2<T>& A){
    return equal(A);
  }
  /*! \brief Overload of the != operator. Just calls `equal` and negates it.
      \param [in] A point to be compared to.
      \returns true if the two configurations are different. 
  */
  bool operator!= (const Point2<T>& A){
    return !equal(A);
  }

  /*! \brief Cast to cv::Point
      \returns The value casted to point
  */
  operator cv::Point() const { 
    return cv::Point(this->x(), this->y()); 
  }

  /**
   * @brief      Overloading of operator <. Since no roght implementetion can be used, then it returns only `true`
   *
   * @param[in]  A     The second `Point2` to be compared to.
   *
   * @return     `true`.
   */
  bool operator<(const Point2<T>& A){ 
    return true;
  }

  /**
   * @brief      Computes the angle between two points, that is the atan of the angular coeficcient of the line joining the two points.
   *
   * @param[in]  P1    The point towards which the line is going.
   * @param[in]  type  The type of the `Angle` to be returned.
   *
   * @tparam     T1    The type of the point.
   *
   * @return     The `Angle`.
   */
  template<class T1>
  Angle th (Point2<T1> P1, 
            Angle::ANGLE_TYPE type=Angle::RAD) const {
    return Angle(atan2((P1.y()-this->y()), (P1.x()-this->x())), type);
  }

  /**
   * @brief      Cast to a Point2 of different type.
   *
   * @tparam     T1    The type of `Point2` to be casted to.
   */
  template<class T1>
  operator Point2<T1>() const {
    return Point2<T1>((T1)(this->x()), (T1)(this->y()));
  }

  /*! \brief Invert the x and y of the point.*/
  void invert (){
    T app=X;
    X=Y;
    Y=app;
  }

};

/*! \brief Transform the angle given i the new reference system where x and y will be swapped.
    \param[in/out] a The angle that need to be inverted.
*/
void invertAngle (Angle & a);

/*!	\brief This class stores a configuration, that is a point and an angle.
		\tparam T1 The type of the coordinates.
*/
template <class T1>
class Configuration2 //: public Tuple<T>
{
private:
  Point2<T1> coord; ///<The coordinate for the configuration.
  Angle th; ///<The angle of the configuration
  
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

  /*! \brief Default constructor that takes the point, the angle, and
            stores them.
      \tparam Type of point in input.
      \param[in] P The coordinates.
      \param[in] _th The angle.
  */
  template<class T2>
  Configuration2( const Point2<T2> P, 
                  const Angle _th) {
    coord=Point2<T1>(P.x(), P.y());
    th=_th;
  }
  
  Point2<T1>  point() const { return coord; } ///<\returns A `Point2` variable containing the coordinates. 
  T1          x()     const { return coord.x(); } ///<\returns The abscissa coordinate.
  T1          y()     const { return coord.y(); } ///<\returns The ordinate coordinate.
  Angle       angle() const { return th; } ///<\returns The angle.
  
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
  
  /**\brief Function to create a stringstream containing the detail of the configuration.
   * \return A stringstream.
   */
  stringstream to_string() const {
    stringstream out;
    out << "x: " << x();
    out << ", y: " << y();
    out << ", th: " << angle();
    return out;
  }

  /*!	\brief Overload of operator << to output the content of a `Configuration2`.
		\param[in] out The output stream.
  	\param[in] data The `Configuration2` to print.
  	\returns An output stream to be printed.
  */
  friend ostream& operator<< (ostream& out, const Configuration2<T1> &data){
    out << data.to_string().str();
    return out;
  }

  /*!\brief Cast of Configuration to Point2.
   *
   * \tparam T2 Type of Point2 to be casted to.
   * \return A Point2 of type T2.
   */
  template<class T2>
  operator Point2<T2> () const {
      return Point2<T2>((T2)(coord.x()), 
                        (T2)(coord.y()));
  }
  
  /*! \brief Copy a configuration into another one.
      \param [in] A Configuration to be coppied.
      \returns this. 
  */
  Configuration2<T1> copy (const Configuration2<T1>& A){
    coord=A.point();
    th=A.angle();
    return *this;
  }
  /*! \brief Overload of the = operatore. Just calls `copy`.
      \param [in] A Configuration to be coppied.
      \returns this. 
  */
  Configuration2<T1> operator= (const Configuration2<T1>& A){
    return this->copy(A);
  }
  /*! \brief Equalize two configurations.
      \param [in] A Configuration to be equalized.
      \returns true if the two configurations are equal. 
  */
  bool equal (const Configuration2<T1>& A){
    return this->angle()==A.angle() && this->point()==A.point();
  }
  /*! \brief Overload of the == operator. Just calls `equal`.
      \param [in] A Configuration to be equalized.
      \returns true if the two configurations are equal. 
  */
  bool operator== (const Configuration2<T1>& A){
    return this->equal(A);
  }

  /*! \brief Overload of the != operator. Just calls `equal` and negates it.
      \param [in] A Configuration to be equalized.
      \returns true if the two configurations are different, false otherwise. 
  */
  bool operator!= (const Configuration2<T1>& A){
    return !this->equal(A);
  }

  /**
   * @brief      Cast a `Configuration2` to a `Point2` of the same type.
   */
  operator Point2<T1>(){
    return this->point();
  }

  /**
   * @brief      Cast a `Configuration2` to a `Configuration2` of a different type.
   *
   * @tparam     T2   The type of the `Configuration2` to be casted to.
   */
  template<class T2>
  operator Configuration2<T2>() const {
    return ( Configuration2<T2>((T2)(this->x()), (T2)(this->y()), this->angle()) );
  }

  /*! \brief Invert the x and y of the point, and even the angle of the configuration.*/
  void invert(){
    coord.invert();
    invertAngle(th);
  }

  // ~Configuration2(){delete coord;}
};

#endif
