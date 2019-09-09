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

	vector<Point2<int> > pathPoints = Planning::planning(img);
	
	cout << "Creating map" << flush;

	Planning::fromVpToPath(pathPoints, path); //return
	cout << "-> -> -> PlanPath end\n" << flush;
	return(true);
}

bool RobotProject::localize(const Mat & img, vector<double> & state){
	Configuration2<double> c = ::localize(img, true);
	// Configuration2<double> c(0.1, 0.1, M_PI/4);

	state.resize(3);
	state[0] = c.x()/1000.0;
	state[1] = c.y()/1000.0;
	state[2] = c.angle().toRad();

	cout << "Fine localize" << flush;
	// mywaitkey();
	return(true);
}