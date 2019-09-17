#include "robotProject.hh"

Settings *sett=new Settings("./exam/data/");

/*! \brief A not used constructor of the class. 

	\param[in] 
	\returns 
*/
// RobotProject::RobotProject(int argc, char* argv[]){
// 	cout << "-> -> -> RobotProject constructor\n";

// 	// cout << "calibration" << endl;
// 	// calibration(); //BUG????!?!?!?!?!?!??!?!?!

// 	cout << endl <<"Configure" << endl;
// 	// configure(true);
// 	cout << "configure done\n";
// }

/*! \brief The main constructor of the class.

	\param[in] camera It is the camera from which the image will be loaded.
	\param[in] frame_time The index of the frame as last one from the camera.
	\returns 
*/
RobotProject::RobotProject(CameraCapture* camera, double& frame_time){
	cout << "-> -> -> RobotProject constructor\n";
	
	sett->cleanAndRead("./exam/data/settings.xml");

	#ifdef CONFIGURE
		Mat img;
		camera->grab(img, frame_time);
		//TODo Remove this.
		// img=imread(sett->unMaps(0).get(0));
		// my_imshow("configure", img);
		// mywaitkey();
		cout << endl << "Configure" << endl;
		configure(img, true);
		cout << "configure done\n";
	#else
		cout << "Configure skipped." << endl;
	#endif
}

/*! \brief The destructor of the class. */
RobotProject::~RobotProject(){
	delete sett;
}

/*! \brief Taken an image this function elaborate it in order to detect the foundamental elements and store them on files.

	\param[in] img The immage that will be processed.
	\returns A true value if everything goes well. False otherwise.
*/
bool RobotProject::preprocessMap(const Mat & img){
	cout << "-> -> -> PreprocessMap\n" << flush;

	Mat internalImg = img;
	cout << endl << "unwrapping" << endl << flush;
	unwrapping(false, &internalImg);

	cout << endl << "detection" << endl << flush;
	detection(false, &internalImg);

	return(true);
}

/*! \brief Taken an image this function try to compute (and return) a path on it, that will bring the robot from its actual position through all the victims and in the end up to the gate.

	\param[in] img The immage that will be processed.
	\param[out] path The path that acts as the return value of the function.
	\returns A true value if everything goes well. False otherwise.
*/
bool RobotProject::planPath(const Mat & img, Path & path){
	cout << "-> -> -> PlanPath\n" << flush;

	vector<Configuration2<double> > pathPoints = Planning::planning(img);
	
	cout << "Creating map\n" << flush;

	Planning::fromVcToPath(pathPoints, path); //return
	cout << "-> -> -> PlanPath end\n" << flush;
	return(true);
}

/*! \brief Taken an image this function try to localize the position and the orientation of the robot. It also apply the neccessary transformation matrix to solve the problem of the different planes.

	\param[in] img The immage that will be processed.
	\param[out] state The state that acts as the return value of the function.
	\returns A true value if everything goes well. False otherwise.
*/
bool RobotProject::localize(const Mat & img, vector<double> & state){
	pair<Configuration2<double>, Configuration2<double> > p = ::localize(img, true);
	Configuration2<double> c = p.second;

	state.resize(3);
	state[0] = c.x()/1000.0;
	state[1] = c.y()/1000.0;
	state[2] = c.angle().toRad();

	return(true);
}