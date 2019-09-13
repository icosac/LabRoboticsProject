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

	#ifdef CONFIGURE
		Mat img;
		camera->grab(img, frame_time);
		
		cout << endl <<"Configure" << endl;
		configure(img, true);
		cout << "configure done\n";
	#else
		cout << "Configure skipped." << endl;
	#endif
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

	vector<Configuration2<double> > pathPoints = Planning::planning(img);
	
	cout << "Creating map" << flush;

	Planning::fromVcToPath(pathPoints, path); //return
	cout << "-> -> -> PlanPath end\n" << flush;
	return(true);
}

bool RobotProject::localize(const Mat & img, vector<double> & state){
	pair<Configuration2<double>, Configuration2<double> > p = ::localize(img, true);
	Configuration2<double> c = p.second;

	state.resize(3);
	state[0] = c.x()/1000.0;
	state[1] = c.y()/1000.0;
	state[2] = c.angle().toRad();

	return(true);
}