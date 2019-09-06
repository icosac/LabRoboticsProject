#include "robotProject.hh"

Settings *sett=new Settings("./exam/data/");

RobotProject::RobotProject(int argc, char* argv[]){
	cout << "-> -> -> RobotProject constructor\n";

	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!

	cout << endl <<"Configure" << endl;
	// configure(true);
	cout << "configure done\n";
}

RobotProject::RobotProject(CameraCapture* camera, double& frame_time){
	cout << "-> -> -> RobotProject constructor\n";
	
	sett->cleanAndRead("./exam/data/settings.xml");
	cout << *sett << endl;

	Mat img;
	camera->grab(img, frame_time);
	
	cout << endl <<"Configure" << endl;
	configure(img, true);
	cout << "configure done\n";
}


RobotProject::~RobotProject(){
	delete sett;
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


bool RobotProject::planPath(const Mat & img, Path & path){
	cout << "-> -> -> PlanPath\n" << flush;

	/*/
	pair< vector<Point2<int> >, Mapp * > tmpPair = planning(img);
	vector<Point2<int> > pathPoints = tmpPair.first;
	Mapp * map = tmpPair.second;

	Mat imageMap = map->createMapRepresentation();

	map->imageAddPoints(imageMap, pathPoints);
	map->imageAddSegments(imageMap, pathPoints);
	delete map;

	#ifdef WAIT
		namedWindow("Map", WINDOW_NORMAL);
		imshow("Map", imageMap);
		mywaitkey();
	#endif
	/*/

	vector<Point2<int> > pathPoints;
	pathPoints.push_back(Point2<int>(100, 100));
	pathPoints.push_back(Point2<int>(200, 200));
	pathPoints.push_back(Point2<int>(300, 300));
	pathPoints.push_back(Point2<int>(400, 400));
	pathPoints.push_back(Point2<int>(500, 500));
	pathPoints.push_back(Point2<int>(600, 600));
	pathPoints.push_back(Point2<int>(700, 700));
	pathPoints.push_back(Point2<int>(800, 800));
	pathPoints.push_back(Point2<int>(900, 900));
	pathPoints.push_back(Point2<int>(1000, 1000));
	//*/
	fromVpToPath(pathPoints, path); //return
	cout << "-> -> -> PlanPath end\n" << flush;
	return(true);
}


bool RobotProject::localize(const Mat & img, vector<double> & state){
	// Configuration2<double> c = ::localize(img, true);
	Configuration2<double> c(0.1, 0.1, M_PI/4);

	state.resize(3);
	state[0] = c.x();
	state[1] = c.y();
	state[2] = c.angle().toRad();

	// mywaitkey();
	return(true);
}