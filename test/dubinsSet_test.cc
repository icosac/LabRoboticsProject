#include<iostream>
#include<dubins.hh>

#define ROB_PIECE_LENGTH 0.5
#define KMAX 0.01

int main (){
	Configuration2<double> start(0.0, 0.0, Angle(M_PI/2.0, Angle::RAD));
	Configuration2<double> end(200.0, 0.0, Angle(3.0*M_PI/2.0, Angle::RAD));
	Configuration2<double> start_(300.0, 0.0, Angle(3.0*M_PI/2.0, Angle::RAD));
	Configuration2<double> end_(500.0, 0.0, Angle(M_PI/2.0, Angle::RAD));

	Dubins<double> A(start, end, KMAX);
	Dubins<double> B(start_, end_, KMAX);

	cout << "A: " << endl << A << endl << endl;
	cout << "B: " << endl << B << endl << endl;

	DubinsSet<double> DS;
	cout << "DS: " << endl << DS << endl << endl;

	DS.addDubins(&A);
	cout << "DS: " << endl << DS << endl << endl;
	DS.addDubins(&B);
	cout << "DS: " << endl << DS << endl << endl;

	return 0;
}



