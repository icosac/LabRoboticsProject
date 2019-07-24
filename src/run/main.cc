// the core of the project

#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>
#include <planning.hh>
#include <configure.hh>


#include<iostream>
using namespace std;

int main (){
	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!
	cout <<"Configure" << endl;
	configure(true);

	cout << "unwrapping" << endl;
	unwrapping();

	cout << "detection" << endl;
	detection();

	cout << "planning" << endl;

	pair< vector<Point2<int> >, Mat > tmpPair = planning();
	vector<Point2<int> > pathPoints = tmpPair.first;
	Mat imageMap = tmpPair.second;

    namedWindow("Map", WINDOW_NORMAL);
	imshow("Map", imageMap);
	waitKey();

	// the robot starts to move MAYBE

	cout << "\nend\n\n";
	return(0);
}
