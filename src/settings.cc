#include"settings.hh"

#define NPOS string::npos ///<Shortcut for string::npos

/*!\brief Function to get all files in directory.
 * From https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
 * @param Path The path to check.
 * @return A vector containing the names of the files in the directory.
 */
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
	  cerr << "Could not open dir: " << path << ".";
	}
	return files;
}

/*!\brief Constructor of class Settings. The value are all set by default. The constructor does NOT read from or write to file.
 *
 * @param mapsFolder A string containing the path for mapsFolder. No certainty is given about the form of this string
 * @param _templatesFolder A String containing the path of the folder containing the number templates.
 * @param _mapsNames A Tuple containing the names of the maps. These are not paths but just names.
 * @param _mapsUnNames A Tuple containing the names of the undistorted maps. These are not paths but just names.
 * @param _calibrationFile A string containing the path to the file containing the data for the calibration.
 * @param _intrinsicCalibrationFile A string containing the path to the file containing the values of the matrix for the calibration.
 * @param _blackMask Filter for black.
 * @param _redMask Filter for red.
 * @param _greenMask Filter for green.
 * @param _victimMask Filter for the victims.
 * @param _blueMask Filter for blue.
 * @param _whiteMask Filter for white.
 * @param _robotMask Filter for the triangle above the robot.
 * @param _kernelSide
 * @param _convexHullFile A String containing the path to file containing the points of the elements in the arena.
 * @param _templates A Tuple containing the names of the templates. These are not paths but just names.
 */
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
      Filter _robotMask,
			int _kernelSide,
			string _convexHullFile,
			vector<string> _templates)
{
	save(	_mapsFolder, _templatesFolder, _mapsNames, _mapsUnNames, 
			_intrinsicCalibrationFile, _calibrationFile, _blackMask, _redMask, _greenMask, 
			_victimMask, _blueMask, _whiteMask, _robotMask, _kernelSide, _convexHullFile, _templates);
}

/*!\brief Function to change values. The value are all set by default. This function does NOT read from or write to file.
 *
 * @param mapsFolder A string containing the path for mapsFolder. No certainty is given about the form of this string
 * @param _templatesFolder A String containing the path of the folder containing the number templates.
 * @param _mapsNames A Tuple containing the names of the maps. These are not paths but just names.
 * @param _mapsUnNames A Tuple containing the names of the undistorted maps. These are not paths but just names.
 * @param _calibrationFile A string containing the path to the file containing the data for the calibration.
 * @param _intrinsicCalibrationFile A string containing the path to the file containing the values of the matrix for the calibration.
 * @param _blackMask Filter for black.
 * @param _redMask Filter for red.
 * @param _greenMask Filter for green.
 * @param _victimMask Filter for the victims.
 * @param _blueMask Filter for blue.
 * @param _whiteMask Filter for white.
 * @param _robotMask Filter for the triangle above the robot.
 * @param _kernelSide
 * @param _convexHullFile A String containing the path to file containing the points of the elements in the arena.
 * @param _templates A Tuple containing the names of the templates. These are not paths but just names.
 */
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
      Filter _robotMask,
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

	//Set all other values
	this->intrinsicCalibrationFile=_intrinsicCalibrationFile;
	this->calibrationFile=_calibrationFile;
	this->blackMask=_blackMask;
	this->redMask=_redMask;
	this->greenMask=_greenMask;
	this->victimMask=_victimMask;
	this->blueMask=_blueMask;
	this->whiteMask=_whiteMask;
  this->robotMask=_robotMask,
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

/*!Writes a vector to a file.
 *
 * @param fs The FileStorage where to write the vector.
 * @param x The vector to write.
 */
inline void vecToFile (FileStorage& fs, vector<int> x){
	for (auto el : x){ 
		fs << el;
	}
}

/*!\brief Function to write settings to file. Default is data/settings.xml.
 *
 * @param _path The path of the file to write to.
 */
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
  fs << NAME(robotMask) << "["; vecToFile(fs, (vector<int>)robotMask); fs <<"]";

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

/*! \brief Function to read from file. The data found is going to be added to the settings. Default file is data/settings.xml
 *
 * @param _path The path of file to read from.
 */
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

  for (uint i=0; i<fs["robotMask"].size(); i++){
    filter.push_back((int)fs["robotMask"][i]);
  }
  robotMask=Filter(filter); filter.clear();

	for (uint i=0; i<fs["victimMask"].size(); i++){
		filter.push_back((int)fs["victimMask"][i]);
	}	
	victimMask=Filter(filter); filter.clear();
		
	kernelSide=fs["kernelSide"];
	convexHullFile=(string)fs["convexHullFile"];

	templatesFolder=(string)fs["templatesFolder"];	
	for (uint i=0; i<fs["templates"].size(); i++){
		string app=(string)fs["templates"][i];
		if (app!=""){
		  templates.add(app);
		}
	}

	fs.release();
}

/*! \brief Function to clean all settings: number types are set to 0, string are set to "", Tuples are set to Tuple<>() and Filter are set to all 0s.
 *
 */
void Settings::clean(){
	this->mapsFolder="";
	this->templatesFolder="";
	this->mapsNames=Tuple<string>();
	this->mapsUnNames=Tuple<string>();
	this->intrinsicCalibrationFile="";
	this->calibrationFile="";
	this->blackMask=Filter();
	this->redMask=Filter();
	this->greenMask=Filter();
	this->victimMask=Filter();
	this->blueMask=Filter();
  this->whiteMask=Filter();
  this->robotMask=Filter();
	this->kernelSide=0;
	this->convexHullFile="";
	this->templates=Tuple<string>();
}

/*! \brief Function to clean all settings and then read from file. Default is data/settings.xml.
 *
 */
void Settings::cleanAndRead(string _path){
	this->clean();
	this->readFromFile(_path);
}


/*!\brief A function to return the paths of a given Tuple of maps.
 *
 * @param _mapNames A Tuple containing the names of the maps to check in the Tuple.
 * @return The paths to the maps if they are found, an empty Tuple otherwise.
 */
Tuple<string> Settings::maps(Tuple<string> _mapNames){
  Tuple<string> ret=Tuple<string>();
  for (auto map : _mapNames){
    string value=maps(map);
    if (value!=""){
      ret.add(value);
    }
  }
  return ret;
}

/*!\brief A function to return the path of a given map.
 *
 * @param _mapName The name of the map to check in the Tuple.
 * @return The path to the map if the map is found, an empty string otherwise.
 */
string Settings::maps(string _mapName){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");

  for (auto map : this->mapsNames){
    if (map==_mapName){
      return prefix+map;
    }
  }
  return string("");
}

/*!\brief Function to return the paths of maps. If ids are not specified all maps are returned.
 *
 * @param ids A Tuple containing the ids (that is the positions in this.mapsNames) of the maps to be retrieved.
 * @return A Tuple containing the paths of the maps.
 */
Tuple<string> Settings::maps(Tuple<int> ids){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
  if (ids.size()==0){
    for (auto Map : this->mapsNames){
      v.add(prefix+Map);
    }
  }
  else {
    for (auto id : ids) {
      v.add(prefix + this->mapsNames.get(id));
    }
  }
  return v;
}

/*!\brief Function to return the path of a map. If id is not specified all maps are returned.
 *
 * @param id A the positions in this.mapsNames of the map to be retrieved
 * @return A Tuple containing the paths of the maps.
 */
Tuple<string> Settings::maps(int id){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
  if (id<0){
    for (auto Map : this->mapsNames){
      v.add(prefix+Map);
    }
  }
  else {
      v.add(prefix + this->mapsNames.get(id));
  }
  return v;
}

/*!\brief A function to return the paths of a given Tuple of undistorted maps.
 *
 * @param _unMapNames A Tuple containing the names of the undistorted maps to check in the Tuple.
 * @return The paths to the undistorted maps if they are found, an empty Tuple otherwise.
 */
Tuple<string> Settings::unMaps(Tuple<string> _unMapNames){
  Tuple<string> ret=Tuple<string>();
  for (auto unMap : _unMapNames){
    string value=unMaps(unMap);
    if (value!=""){
      ret.add(value);
    }
  }
  return ret;
}

/*!\brief A function to return the path of a given undistorted map.
 *
 * @param _unMapName The name of the undistorted map to check in the Tuple.
 * @return The path to the undistorted map if it is found, an empty string otherwise.
 */
string Settings::unMaps(string _unMapName){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");

  for (auto unMap : this->mapsUnNames){
    if (unMap==_unMapName){
      return prefix+unMap;
    }
  }
  return string("");
}

/*!\brief Function to return the paths of undistorted maps. If ids are not specified all undistorted maps are returned.
 *
 * @param ids A Tuple containing the ids (that is the positions in this.mapsUnNames) of the undistorted maps to be retrieved.
 * @return A Tuple containing the paths of the undistorted maps.
 */
Tuple<string> Settings::unMaps(Tuple<int> ids){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
  if (ids.size()==0){
    for (auto unMap : this->mapsUnNames){
      v.add(prefix+unMap);
    }
  }
  else {
    for (auto id : ids) {
      v.add(prefix + this->mapsUnNames.get(id));
    }
  }
  return v;
}

/*!\brief Function to return the path of an undistorted map. If id is not specified all undistorted maps are returned.
 *
 * @param id A the positions in this.mapsUnNames of the undistorted map to be retrieved
 * @return A Tuple containing the paths of the undistorted maps.
 */
Tuple<string> Settings::unMaps(int id){
  string prefix=this->mapsFolder+(this->mapsFolder.back()=='/' ? "" : "/");
  Tuple<string> v;
  if (id<0){
    for (auto unMap : this->mapsUnNames){
      v.add(prefix+unMap);
    }
  }
  else {
      v.add(prefix + this->mapsUnNames.get(id));
  }
  return v;
}

/*!\brief Change the values of Tuple of filters. Mind that no write function is called.
 *
 * @param color A Tuple containing the colors of the filters to change.
 * @param fil The new filters to be stored.
 */
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

/*!\brief Change the values of a filter. Mind that no write function is called.
 *
 * @param color The filter to change.
 * @param fil The new filter to be stored.
 */
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
