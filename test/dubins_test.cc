#include "../src/dubins.hh"

int main (){
	Dubins<double> d=Dubins<double>(
			Configuration2<double>(0, 0, Angle((-2.0/3.0)*M_PI, Angle::RAD)),
			Configuration2<double>(4, 0, Angle(M_PI/3.0, Angle::RAD))
		);
	cout << d << endl;
	return 0;
}