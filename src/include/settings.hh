#ifndef SETTINGS_HH
#define SETTINGS_HH

//TODO change maths.hh

#include <opencv2/core/core.hpp>
#include "filter.hh"
#include <iostream>
#include "maths.hh"
#include <string>
#include <dirent.h>
#include <sstream>

using namespace cv;
using namespace std;

#define NAME(x) #x

class Settings{
public:
	string mapsFolder;
	Tuple<string> mapsNames;
	Tuple<string> mapsUnNames;
	string calibrationFile;
	Filter blackMask;
	Filter redMask;
	Filter greenMask;
	Filter victimMask;
	Filter blueMask;
	Filter whiteMask;
	int kernelSide;
	string convexHullFile;
	string templatesFolder;
	Tuple<string> templates;

	Settings(
			string mapsFolder="data/map",
			string _templatesFolder="data/num_template/",
			vector<string> _mapsNames={},
			vector<string> _mapsUnNames={},
			string _calibrationFile="data/intrinsic_calibration.xml",
			Filter _blackMask=Filter(0, 0, 0, 179, 255, 70),
			Filter _redMask=Filter(15, 100, 140, 160, 255, 255),
			Filter _greenMask=Filter(54, 74, 25, 119, 255, 88),
			Filter _victimMask=Filter(0, 0, 0, 179, 255, 80),
			Filter _blueMask=Filter(100, 100, 40, 140, 200, 170),
			Filter _whiteMask=Filter(100, 100, 40, 140, 200, 170),
			int _kernelSide=9,
			string _convexHullFile="data/convexHull.xml",
			vector<string> _templates={}
	);

	~Settings();	

	void save (
			string mapsFolder="data/map",
			string _templatesFolder="data/num_template/",
			vector<string> _mapsNames={},
			vector<string> _mapsUnNames={},
			string _calibrationFile="data/intrinsic_calibration.xml",
			Filter _blackMask=Filter(0, 0, 0, 179, 255, 70),
			Filter _redMask=Filter(15, 100, 140, 160, 255, 255),
			Filter _greenMask=Filter(54, 74, 25, 119, 255, 88),
			Filter _victimMask=Filter(0, 0, 0, 179, 255, 80),
			Filter _blueMask=Filter(100, 100, 40, 140, 200, 170),
			Filter _whiteMask=Filter(100, 100, 40, 140, 200, 170),
			int _kernelSide=9,
			string _convexHullFile="data/convexHull.xml",
			vector<string> _templates={}
	);

	void writeToFile(string _path="data/settings.xml");
	void readFromFile(string _path="data/settings.xml");

	enum COLOR {BLACK, RED, GREEN, VICTIMS, BLUE, WHITE};


	void changeMask(Tuple<COLOR> color, Tuple<Filter> fil);
	void changeMask(COLOR color, Filter fil);

	stringstream to_string () const {
		stringstream out;
		out << NAME(mapsNames) << ": " << mapsNames << endl;
		out << NAME(mapsUnNames) << ": " << mapsUnNames << endl;
		out << NAME(calibrationFile) << ": " << calibrationFile << endl;
		out << NAME(blackMask) << ": " << blackMask << endl;
		out << NAME(redMask) << ": " << redMask << endl;
		out << NAME(greenMask) << ": " << greenMask << endl;
		out << NAME(blueMask) << ": " << blueMask << endl;
		out << NAME(whiteMask) << ": " << whiteMask << endl;
		out << NAME(victimMask) << ": " << victimMask << endl;
		out << NAME(kernelSide) << ": " << kernelSide << endl;
		out << NAME(templatesFolder) << ": " << templatesFolder << endl;
		out << NAME(convexHullFile) << ": " << convexHullFile << endl;
		out << NAME(templates) << ": " << templates << endl;
		return out;
	}


	/*! This function overload the << operator so to print with `std::cout` the most essential info, that is the dimension and the type of angle.
			\param[in] out The out stream.
			\param[in] data The angle to print.
			\returns An output stream to be printed.
	*/
	friend ostream& operator<<(ostream &out, const Settings& data) {
		out << data.to_string().str();
		return out;
	}

};
#endif