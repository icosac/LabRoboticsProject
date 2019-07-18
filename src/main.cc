// the core of the project

#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>

#include<iostream>
using namespace std;

int main (){
	cout << "calibration" << endl;
	calibration();
	cout << "unwrapping" << endl;
	unwrapping();
	cout << "detection" << endl;
	detection();

	// cout << "planning" << endl;
	// planning();

	// the robot start to move

	cout << "\nend\n\n";

	return 0;
}
