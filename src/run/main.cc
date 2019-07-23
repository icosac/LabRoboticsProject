// the core of the project

#include "detection.hh"
#include "unwrapping.hh"
#include "calibration.hh"
#include "planning.hh"

#include<iostream>
using namespace std;

int main (){
	cout << "calibration" << endl;
	// calibration();

	cout << "unwrapping" << endl;
	unwrapping();

	cout << "detection" << endl;
	detection();

	cout << "planning" << endl;
	planning();

	// pair< vector<Point2<int> >, Mat > tmpPair = planning();
	// vector<Point2<int> > pathPoints = tmpPair.first;
	// Mat imageMap = tmpPair.second;

    // namedWindow("Map", WINDOW_AUTOSIZE);
	// imshow("Map", imageMap);



	// the robot start to move

	cout << "\nend\n\n";
	return(0);
}
