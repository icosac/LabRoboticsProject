#include"settings.hh"

#define NPOS string::npos

vector<string> getFiles(const string& path){
	vector<string> files;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (path.c_str())) != NULL) {
	  /* print all the files and directories within directory */
	  while ((ent = readdir (dir)) != NULL) {
	  	string file=string(ent->d_name);
	    if (!(file=="." || file=="..")){
	    	files.push_back(file);
	    }
	  }
	  closedir (dir);
	} else {
	  /* could not open directory */
	  perror (("Could not open dir: "+path+".").c_str());
	}
	return files;
}

Settings::Settings(
			string _mapFolder,
			string _templatesFolder,
			vector<string> _mapNames,
			vector<string> _mapUnNames,
			string _calibrationFile,
			Filter _blackMask,
			Filter _redMask,
			Filter _greenMask,
			Filter _victimMask,
			Filter _blueMask,
			int _kernelSide,
			string _convexHullFile,
			vector<string> _templates)
{
	//Get Maps
	for (auto el : _mapNames){
		this->mapNames.add(el);
	}
	for (auto el : _mapUnNames){
		this->mapUnNames.add(el);
	}
	for (auto el : getFiles(_mapFolder)){
		if (el.find("UN")==NPOS){ 	//Get distorted maps
			this->mapNames.add(el);
		}
		else { //Get undistorted maps
			this->mapUnNames.add(el);
		}
	}

	//Set all other values
	this->calibrationFile=_calibrationFile;
	this->blackMask=_blackMask;
	this->redMask=_redMask;
	this->greenMask=_greenMask;
	this->victimMask=_victimMask;
	this->blueMask=_blueMask;
	this->kernelSide=_kernelSide;
	this->convexHullFile=_convexHullFile;

	//Get templates
	this->templatesFolder=_templatesFolder;
	for (auto el : _templates){
		this->templates.add(el);
	}

	for (auto el : getFiles(_templatesFolder)){
		this->templates.add(el);
	}
}

