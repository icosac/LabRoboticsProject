#include "robotProject.hh"

CameraCapture* camera;

RobotProject::RobotProject(){
	cout << "-> -> -> RobotProject constructor\n";
	
	// cout << "calibration" << endl;
	// calibration(); //BUG????!?!?!?!?!?!??!?!?!

	//Throw away first n frames to calibrate camera
	CameraCapture::input_options_t options(1080, 1920, 30, 0);
    camera= new CameraCapture(options);

    double frame_time=0.0;
    for (int i=0; i<50; i++){
        Mat frame;
        camera->grab(frame, frame_time);
	    COUT(frame_time)
    }
	cout << endl <<"Configure" << endl;
	configure(true);
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