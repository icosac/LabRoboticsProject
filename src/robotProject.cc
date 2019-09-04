#include "robotProject.hh"

RobotProject::RobotProject(){
	cout << "-> -> -> RobotProject constructor\n";

	sett->cleanAndRead();
	
	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!

	cout << endl <<"Configure" << endl;
	configure(false);
}

bool RobotProject::preprocessMap(const Mat & img){
	cout << "-> -> -> PreprocessMap\n" << flush;

	Mat internalImg = img;
	cout << endl << "unwrapping" << endl << flush;
	unwrapping(false, &internalImg);

	cout << endl << "detection" << endl << flush;
	detection(false, &internalImg);

	return(true);
}


bool RobotProject::planPath(const Mat & img, ClipperLib::Path & path){
	cout << "-> -> -> PlanPath\n" << flush;

	pair< vector<Point2<int> >, Mapp * > tmpPair = planning(img);
	vector<Point2<int> > pathPoints = tmpPair.first;
	Mapp * map = tmpPair.second;

	Mat imageMap = map->createMapRepresentation();

	map->imageAddPoints(imageMap, pathPoints);
	map->imageAddSegments(imageMap, pathPoints);
	delete map;

    fromVpToPath(pathPoints, path); //return

	#ifdef WAIT
		namedWindow("Map", WINDOW_NORMAL);
		imshow("Map", imageMap);
		mywaitkey();
	#endif
	cout << "fine\n";
	return(true);
}


bool RobotProject::localize(const Mat & img, vector<double> & state){
	cout << "-> -> -> localize\n";
	Configuration2<double> c = ::localize(img, true);
	
	state.resize(3);
	state[0] = c.x();
	state[1] = c.y();
	state[2] = c.angle().toRad();

	mywaitkey();
	return(true);
}