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

	map->imageAddPoints(imageMap, pathPoints);
	map->imageAddSegments(imageMap, pathPoints);

	#ifdef WAIT
		namedWindow("Map", WINDOW_NORMAL);
		imshow("Map", imageMap);
		mywaitkey();
	#endif

	// the robot starts to move MAYBE
	Point2<int> p;
	for(int i=0; i<100; i++){	// 100*(<50ms) = (<5sec)
		p = localize();
	}

	cout << "\nend\n\n";
	return(0);
}
