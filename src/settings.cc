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
			string _mapsFolder,
			string _templatesFolder,
			vector<string> _mapsNames,
			vector<string> _mapsUnNames,
			string _intrinsicCalibrationFile,
			string _calibrationFile,
			Filter _blackMask,
			Filter _redMask,
			Filter _greenMask,
			Filter _victimMask,
			Filter _blueMask,
			Filter _whiteMask,
			int _kernelSide,
			string _convexHullFile,
			vector<string> _templates)
{
	save(	_mapsFolder, _templatesFolder, _mapsNames, _mapsUnNames, 
			_intrinsicCalibrationFile, _calibrationFile, _blackMask, _redMask, _greenMask, 
			_victimMask, _blueMask, _whiteMask, _kernelSide, _convexHullFile, _templates);
}

void Settings::save (
			string _mapsFolder,
			string _templatesFolder,
			vector<string> _mapsNames,
			vector<string> _mapsUnNames,
			string _intrinsicCalibrationFile,
			string _calibrationFile,
			Filter _blackMask,
			Filter _redMask,
			Filter _greenMask,
			Filter _victimMask,
			Filter _blueMask,
			Filter _whiteMask,
			int _kernelSide,
			string _convexHullFile,
			vector<string> _templates)
{
	//Get Maps
	this->mapsFolder=_mapsFolder;
	for (auto el : _mapsNames){
		this->mapsNames.add(el);
	}
	for (auto el : _mapsUnNames){
		this->mapsUnNames.add(el);
	}
	for (auto el : getFiles(_mapsFolder)){
		if (el.find("UN")==NPOS){ 	//Get distorted maps
			this->mapsNames.add(el);
		}
		else { //Get undistorted maps
			this->mapsUnNames.add(el);
		}
	}
	cout << "Maps: ";
	for (int i=0; i< this->mapsNames.size(); i++){
		cout << this->mapsNames.get(i) << " ";
	}
	cout << endl;

	//Set all other values
	this->intrinsicCalibrationFile=_intrinsicCalibrationFile;
	this->calibrationFile=_calibrationFile;
	this->blackMask=_blackMask;
	this->redMask=_redMask;
	this->greenMask=_greenMask;
	this->victimMask=_victimMask;
	this->blueMask=_blueMask;
	this->whiteMask=_whiteMask;
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

inline void vecToFile (FileStorage& fs, vector<int> x){
	for (auto el : x){ 
		fs << el;
	}
}

void Settings::writeToFile(string _path){
	FileStorage fs(_path, FileStorage::WRITE);
	
	fs << NAME(mapsFolder) << mapsFolder;

	fs << NAME(mapsNames) << "[";
	for (int i=0; i<mapsNames.size(); i++){
		fs << mapsNames.get(i);
	}
	fs << "]";
	
	fs << NAME(mapsUnNames) << "[";
	for (int i=0; i<mapsUnNames.size(); i++){
		fs << mapsUnNames.get(i);
	}
	fs << "]";
	
	fs << NAME(intrinsicCalibrationFile) << intrinsicCalibrationFile;
	fs << NAME(calibrationFile) << calibrationFile;
	fs << NAME(blackMask) << "["; vecToFile(fs, (vector<int>)blackMask); fs <<"]";
	fs << NAME(redMask) << "["; vecToFile(fs, (vector<int>)redMask); fs <<"]";
	fs << NAME(greenMask) << "["; vecToFile(fs, (vector<int>)greenMask); fs <<"]";
	fs << NAME(blueMask) << "["; vecToFile(fs, (vector<int>)blueMask); fs <<"]";
	fs << NAME(whiteMask) << "["; vecToFile(fs, (vector<int>)whiteMask); fs <<"]";
	fs << NAME(victimMask) << "["; vecToFile(fs, (vector<int>)victimMask); fs <<"]";
	
	fs << NAME(kernelSide) << kernelSide;
	fs << NAME(convexHullFile) << convexHullFile;
	fs << NAME(templatesFolder) << templatesFolder;
	
	fs << NAME(templates) << "["; 

	for (int i=0; i<templates.size(); i++){
		fs << templates.get(i);
	}
	fs << "]";
	fs.release();
}

void Settings::readFromFile(string _path){
	FileStorage fs(_path, FileStorage::READ);

	mapsFolder=(string)fs["mapsFolder"];
	for (uint i=0; i<fs["mapsNames"].size(); i++){
		mapsNames.add((string)fs["mapsNames"][i]);
	}
	for (uint i=0; i<fs["mapsUnNames"].size(); i++){
		mapsUnNames.add((string)fs["mapsUnNames"][i]);
	}

	intrinsicCalibrationFile=(string)fs["intrinsicCalibrationFile"];
	calibrationFile=(string)fs["calibrationFile"];
	

	vector<int> filter;
	for (uint i=0; i<fs["blackMask"].size(); i++){
		filter.push_back((int)fs["blackMask"][i]);
	}	
	blackMask=Filter(filter); filter.clear();

	for (uint i=0; i<fs["redMask"].size(); i++){
		filter.push_back((int)fs["redMask"][i]);
	}	
	redMask=Filter(filter); filter.clear();
	
	for (uint i=0; i<fs["greenMask"].size(); i++){
		filter.push_back((int)fs["greenMask"][i]);
	}	
	greenMask=Filter(filter); filter.clear();
	
	for (uint i=0; i<fs["blueMask"].size(); i++){
		filter.push_back((int)fs["blueMask"][i]);
	}	
	blueMask=Filter(filter); filter.clear();
	
	for (uint i=0; i<fs["whiteMask"].size(); i++){
		filter.push_back((int)fs["whiteMask"][i]);
	}	
	whiteMask=Filter(filter); filter.clear();

	for (uint i=0; i<fs["victimMask"].size(); i++){
		filter.push_back((int)fs["victimMask"][i]);
	}	
	victimMask=Filter(filter); filter.clear();
		
	kernelSide=fs["kernelSide"];
	convexHullFile=(string)fs["convexHullFile"];
	
	templatesFolder=(string)fs["templatesFolder"];	
	for (uint i=0; i<fs["templates"].size(); i++){
		string app=(string)fs["templates"];
		if (app!=""){
		  templates.add(app);
		}
	}

	fs.release();
}


Tuple<string> Settings::maps(Tuple<string> _mapNames){
  Tuple<string> ret;
  for (auto map : _mapNames){
    string value=maps(map);
    if (value!=""){
      ret.add(value);
    }
  }
  return ret;
}

string Settings::maps(string _mapName){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");

  for (auto map : this->mapsNames){
    if (map==_mapName){
      return prefix+map;
    }
  }
  return string("");
}


Tuple<string> Settings::maps(int id){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
    if (id<0){
      for (auto map : this->mapsNames){
        v.add(prefix+map);
      }
    }
    else {
      v.add(prefix+this->mapsNames.get(id));
    }
  return v;
}

Tuple<string> Settings::unMaps(Tuple<string> _unMapNames){
  Tuple<string> ret;
  for (auto unMap : _unMapNames){
    string value=unMaps(unMap);
    if (value!=""){
      ret.add(value);
    }
  }
  return ret;
}

string Settings::unMaps(string _unMapName){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");

  for (auto unMap : this->mapsUnNames){
    if (unMap==_unMapName){
      return prefix+unMap;
    }
  }
  return string("");
}

Tuple<string> Settings::unMaps(int id){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
  if (id<0){
    for (auto unMap : this->mapsUnNames){
      v.add(prefix+unMap);
    }
  }
  else {
    v.add(prefix+this->mapsUnNames.get(id));
  }
  return v;
}

void Settings::changeMask(Tuple<COLOR> color, Tuple<Filter> fil){
	if (color.size()!=fil.size()){
		cout << "Color and filter tuples must have same size." << endl;
	}
	else {
		for (int i=0; i<color.size(); i++){
			changeMask(color.get(i), fil.get(i));
		}
	}
}

void Settings::changeMask(COLOR color, Filter fil){
	switch(color){
		case BLACK:{
			blackMask=fil;
			break;
		}
		case GREEN:{
			greenMask=fil;
			break;
		}
		case RED:{
			redMask=fil;
			break;
		}
		case VICTIMS:{
			victimMask=fil;
			break;
		}
		case BLUE:{
			blueMask=fil;
			break;
		}
		case WHITE:{
			whiteMask=fil;
			break;
		}
		default:{
			cout << "Color is not correct" << endl;
		}
	}
}
