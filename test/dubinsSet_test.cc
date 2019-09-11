#include<iostream>
#include<dubins.hh>

#define ROB_PIECE_LENGTH 0.5

int main (){
	Configuration2<double> start(1.4, 0.0, Angle(0.0, Angle::RAD));
	vector<Point2<double> > intermediates {Point2<double> (10.0, 0.0)};
	Configuration2<double> end(0.0, 0.0, Angle(M_PI, Angle::RAD));

	DubinsSet<double> d (start, end, Tuple<Point2<double> >(intermediates), 0.1);
	cout << d << endl;
	cout << d.getBegin() << endl;
	cout << d.getEnd() << endl;
	
	

	Tuple<Tuple<Tuple<Configuration2<double> > > > t = d.splitIt(ROB_PIECE_LENGTH);
	for (auto a : t){
		cout << a << endl;
	}

	return 0;
}