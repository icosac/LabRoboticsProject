// the core of the project

#include <utils.hh>
#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>
#include <planning.hh>
#include <configure.hh>
#include <settings.hh>

#include<iostream>
using namespace std;

#define WAIT
#define DEBUG

Settings *sett =new Settings();

int main (){
	sett->cleanAndRead();
	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!
	cout << endl <<"Configure" << endl;
	configure(false);

	cout << endl << "unwrapping" << endl;
	unwrapping();

	cout << endl << "detection" << endl;
	detection();

	cout << endl << "planning" << endl;

	pair< vector<Point2<int> >, Mapp * > tmpPair = planning();
	vector<Point2<int> > pathPoints = tmpPair.first;
	Mapp * map = tmpPair.second;

	Mat imageMap = map->createMapRepresentation();

	map->imageAddPoint(imageMap, Point2<int>(100, 150) );
	map->imageAddSegments(imageMap, pathPoints);

	#ifdef WAIT
		namedWindow("Map", WINDOW_NORMAL);
		imshow("Map", imageMap);
		mywaitkey();
	#endif

	// the robot starts to move MAYBE

	cout << "\nend\n\n";
	return(0);
}
