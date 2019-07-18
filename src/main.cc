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
	/*pair< vector<Point2<int> >, Mat > tmpPair = */planning();
	// vector<Point2<int> > pathPoints = tmpPair.first;
	// Mat imageMap = tmpPair.second;

	// the robot start to move

	cout << "\nend\n\n";
	return(0);
}
