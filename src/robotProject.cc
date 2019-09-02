#include "robotProject.hh"

RobotProject::RobotProject(){}

bool RobotProject::preprocessMap(/*const*/Mat & img){
	cout << "\t-> preprocessMap\n";

	return(true);
}


bool RobotProject::planPath(/*const*/ Mat & img, ClipperLib::Path & path){
	cout << "\t-> planPath\n";

	return(true);
}


bool RobotProject::localize(/*const*/ Mat & imgNew, vector<double> & state){
	cout << "\t-> localize\n";

	return(true);
}