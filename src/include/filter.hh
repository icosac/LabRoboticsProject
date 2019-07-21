#ifndef FILTER_HH
#define FILTER_HH

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*! A class to store the values for an HSV filter with lower and higher boundary.
 *
 */
class Filter {
	public:
		int low_h; ///<Lower value for hue
		int low_s; ///<Lower value for saturation
		int low_v; ///<Lower value for value
		int high_h;///<Higher value for hue
		int high_s;///<Higher value for saturation
		int high_v;///<Higher value for value

		/*! \brief Default constructor: it set all values to 0.
		 *
		 */
		Filter () : low_h(0), low_s(0), low_v(0), high_h(0), high_s(0), high_v(0) {}

		/*!\brief Constructor that sets all the values.
		 *
		 * @param _low_h Lower value for hue
		 * @param _low_s Lower value for saturation 
		 * @param _low_v Lower value for value
		 * @param _high_h Higher value for hue 
		 * @param _high_s Higher value for saturation
		 * @param _high_v Higher value for value
		 */
		Filter (int _low_h, int _low_s, int _low_v, int _high_h, int _high_s, int _high_v) :
			low_h(_low_h), low_s(_low_s), low_v(_low_v),
			high_h(_high_h), high_s(_high_s), high_v(_high_v) {}

		/*!\brief Constructor from a vector.
		 *
		 * @param v The vector containing the 6 values. Mind that they must be 6.
		 */
		Filter (vector<int> v){
			if (v.size()!=6){
				cout << "Wrong number of element in vector." << endl;
			}
			else {
				*this=Filter(v[0], v[1], v[2], v[3], v[4], v[5]);
			}
		}


		Scalar Low (){ return Scalar(low_h, low_s, low_v); } ///<Returns a Scalar containing the lower boudary

		Scalar High (){ return Scalar(high_h, high_s, high_v); } ///<Returns a Scalar containing the lower boudary

    /*!\brief Save value in a stringstream.
     *
     * @return A stringstream containing the values of both boundaries.
     */
		stringstream to_string () const {
			stringstream out;
			out << low_h << " " << low_s << " " << low_v << " " << high_h << " " << high_s << " " << high_v;
			return out;
		}

		/*!\brief A function to copy a filter to this.
		 *
		 * @param fil The filter to be copied.
		 * @return this filter with the new values copied.
		 */
		Filter copy (const Filter& fil){
			this->low_h=fil.low_h;
			this->low_s=fil.low_s;
			this->low_v=fil.low_v;
			this->high_h=fil.high_h;
			this->high_s=fil.high_s;
			this->high_v=fil.high_v;
			return *this;
		}

		/*!\brief Overload of operator =. It just calls the copy function.
		 *
		 * @param filt The filter to be copied.
		 * @return this filter with the new values copied.
		 */
		Filter operator= (const Filter& filt){
			return copy(filt);
		}

    /*!\brief Overload of operator cast to vector<int>.
     *
     * @return A vector containing the 6 values.
     */
    operator vector<int>() const{
			vector<int> v={low_h, low_s, low_v, high_h, high_s, high_v};
			return v;
		}

		/*! This function overload the << operator so to print with `std::cout` .
				\param[in] out The out stream.
				\param[in] data The filter to print.
				\returns An output stream to be printed.
		*/
		friend ostream& operator<<(ostream &out, const Filter& data) {
			out << data.to_string().str();
			return out;
		}
};

#endif
