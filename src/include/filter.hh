#ifndef FILTER_HH
#define FILTER_HH

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Filter {
	public:
		int low_h, low_s, low_v;
		int high_h, high_s, high_v;
		
		Filter () : low_h(0), low_s(0), low_v(0), high_h(0), high_s(0), high_v(0) {}

		Filter (int _low_h, int _low_s, int _low_v, int _high_h, int _high_s, int _high_v) :
			low_h(_low_h), low_s(_low_s), low_v(_low_v),
			high_h(_high_h), high_s(_high_s), high_v(_high_v) {}

		Scalar Low (){
			return Scalar(low_h, low_s, low_v);
		}

		Scalar High (){
			return Scalar(high_h, high_s, high_v);
		}

		stringstream to_string () const {
			stringstream out;
			out << low_h << " " << low_s << " " << low_v << " " << high_h << " " << high_s << " " << high_v;
			return out;
		}


		/*! This function overload the << operator so to print with `std::cout` the most essential info, that is the dimension and the type of angle.
				\param[in] out The out stream.
				\param[in] data The angle to print.
				\returns An output stream to be printed.
		*/
		friend ostream& operator<<(ostream &out, const Filter& data) {
			out << data.to_string().str();
			return out;
		}
};

#endif