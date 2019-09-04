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
	unwrapping(true, &internalImg);

	cout << endl << "detection" << endl << flush;
	detection(true, &internalImg);

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
	Point2<int> p = ::localize(img, true);
	mywaitkey('q');
	return(true);
}