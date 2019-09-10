#include<iostream>
#include<dubins.hh>

int main (){
	Configuration2<double> start(0.0, 0.0, Angle(0.0, Angle::RAD));
	vector<Point2<double> > intermediates {Point2<double> (1.0, 0.0)};
	Configuration2<double> end(3.0, 0.0, Angle(0.0, Angle::RAD));

	DubinsSet<double> d (start, end, Tuple<Point2<double> >(intermediates), 2);
	cout << d << endl;
	cout << d.getBegin() << endl;
	cout << d.getEnd() << endl;

	Configuration2<double> start_(1.0, 0.0, Angle(M_PI/2.0, Angle::RAD));
	vector<Point2<double> > intermediates_ {Point2<double> (2.0, 2.0)};
	Configuration2<double> end_(3.0, 0.0, Angle(M_PI/2.0, Angle::RAD));
	DubinsSet<double> e (start_, end_, Tuple<Point2<double> >(intermediates_), 2);

	cout << e << endl;
	d.join(&e);
	cout << d << endl;
	cout << d.getBegin() << endl;
	cout << d.getEnd() << endl;

	return 0;
}