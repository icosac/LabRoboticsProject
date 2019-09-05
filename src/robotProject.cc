#include "robotProject.hh"

Settings *sett = new Settings("./exam/data/");

RobotProject::RobotProject(int argc, char* argv[]){
	cout << "-> -> -> RobotProject constructor\n";
	
	COUT(*sett)

	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!

	cout << endl <<"Configure" << endl;
	// configure(true);
	cout << "configure done\n";
}

RobotProject::RobotProject(CameraCapture* camera, double& frame_time){
	cout << "-> -> -> RobotProject constructor\n";
	
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

	// pair< vector<Point2<int> >, Mapp * > tmpPair = planning(img);
	// vector<Point2<int> > pathPoints = tmpPair.first;
	// Mapp * map = tmpPair.second;

	// Mat imageMap = map->createMapRepresentation();

	// map->imageAddPoints(imageMap, pathPoints);
	// map->imageAddSegments(imageMap, pathPoints);
	// delete map;

	// fromVpToPath(pathPoints, path); //return

	// #ifdef WAIT
	// 	namedWindow("Map", WINDOW_NORMAL);
	// 	imshow("Map", imageMap);
	// 	mywaitkey();
	// #endif

	//Pose(double s, double x, double y, double theta, double kappa)
	path.points.push_back(Pose(0.1, 100, 100, 3.14/4, 0));
	path.points.push_back(Pose(0.2, 200, 200, 3.14/4, 0));
	path.points.push_back(Pose(0.3, 300, 300, 3.14/4, 0));
	path.points.push_back(Pose(0.4, 400, 400, 3.14/4, 0));
	path.points.push_back(Pose(0.5, 500, 500, 3.14/4, 0));
	path.points.push_back(Pose(0.6, 600, 600, 3.14/4, 0));
	path.points.push_back(Pose(0.7, 700, 700, 3.14/4, 0));
	path.points.push_back(Pose(0.8, 800, 800, 3.14/4, 0));
	path.points.push_back(Pose(0.9, 900, 900, 3.14/4, 0));
	path.points.push_back(Pose(1.0, 1000, 1000, 3.14/4, 0));
	return(true);
}


bool RobotProject::localize(const Mat & img, vector<double> & state){
	cout << "-> -> -> localize\n";
	Configuration2<double> c = ::localize(img, true);
	
	state.resize(3);
	state[0] = c.x();
	state[1] = c.y();
	state[2] = c.angle().toRad();

	// mywaitkey();
	return(true);
}