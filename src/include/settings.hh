#ifndef SETTINGS_HH
#define SETTINGS_HH

#include <filter.hh>
#include <maths.hh>
#include <utils.hh>

#include <opencv2/core/core.hpp>

#include <iostream>
#include <string>
#include <dirent.h>
#include <sstream>

using namespace cv;
using namespace std;

#define PATH "./data/"
#define FILE "./data/settings.xml"

/*!
 * Class that stores settings for the projects such as location of files, name of maps and filters to use.
 * Mind that when created it does not read from file by default but the function must be invoked.
 */
class Settings{
public:
	string baseFolder;								///<A string containing the path for the base dir of data.
	string mapsFolder;                ///<A string containing the name for maps folder. No certainty is given about the form of this string
	Tuple<string> mapsNames;          ///<A Tuple containing the names of the maps. These are not paths but just names.
	Tuple<string> mapsUnNames;        ///<A Tuple containing the names of the undistorted maps. These are not paths but just names.
	string intrinsicCalibrationFile;  ///<A string containing the name to the file containing the values of the matrix for the calibration.
	string calibrationFile;           ///<A string containing the name to the file containing the data for the calibration.
	Filter blackMask;                 ///<Filter for black.
	Filter redMask;                   ///<Filter for red.
	Filter greenMask;                 ///<Filter for green.
	Filter victimMask;                ///<Filter for the victims.
	Filter blueMask;                  ///<Filter for blue.
	Filter robotMask;                 ///<Filter for the triangle above the robot.
	int kernelSide;
	string convexHullFile;            ///<AString containing the name to file containing the points of the elements in the arena.
	string templatesFolder;           ///<A String containing the name of the folder containing the number templates.
	Tuple<string> templates;          ///<A Tuple containing the names of the templates. These are not paths but just names.

	/*!\brief Constructor of class Settings. The value are all set by default. The constructor does NOT read from or write to file.
	 *
	 * \param[in] baseFolder A string containing the path for the base dir of data.
	 * \param[in] mapsFolder A string containing the name for maps folder. No certainty is given about the form of this string.
	 * \param[in] _templatesFolder A String containing the name of the folder containing the number templates.
	 * \param[in] _mapsNames A Tuple containing the names of the maps. These are not paths but just names.
	 * \param[in] _mapsUnNames A Tuple containing the names of the undistorted maps. These are not paths but just names.
	 * \param[in] _calibrationFile A string containing the name to the file containing the data for the calibration.
	 * \param[in] _intrinsicCalibrationFile A string containing the name to the file containing the values of the matrix for the calibration.
	 * \param[in] _blackMask Filter for black.
	 * \param[in] _redMask Filter for red.
	 * \param[in] _greenMask Filter for green.
	 * \param[in] _victimMask Filter for the victims.
	 * \param[in] _blueMask Filter for blue.
	 * \param[in] _robotMask Filter for the triangle above the robot.
	 * \param[in] _kernelSide 
	 * \param[in] _convexHullFile A String containing the name to file containing the points of the elements in the arena.
	 * \param[in] _templates A Tuple containing the names of the templates. These are not paths but just names.
	 */
	Settings(
			string _baseFolder="data/",
			string _mapsFolder="map",
			string _templatesFolder="num_template/",
			vector<string> _mapsNames={},
			vector<string> _mapsUnNames={},
			string _intrinsicCalibrationFile="intrinsic_calibration.xml",
			string _calibrationFile="calib_config.xml",
			Filter _blackMask=Filter(0, 0, 0, 179, 255, 70),
			Filter _redMask=Filter(15, 100, 140, 160, 255, 255),
			Filter _greenMask=Filter(54, 74, 25, 119, 255, 88),
			Filter _victimMask=Filter(0, 0, 0, 179, 255, 80),
			Filter _blueMask=Filter(100, 100, 40, 140, 200, 170),
      Filter _roboteMask=Filter(100, 100, 40, 140, 200, 170),
			int _kernelSide=9,
			string _convexHullFile="convexHull.xml",
			vector<string> _templates={}
	);

	/*!
	 * \brief Destructor.
	 */
	~Settings();

  /*!\brief Function to change values. The value are all set by default. This function does NOT read from or write to file.
   *
	 * \param[in] baseFolder A string containing the path for the base dir of data.
   * \param[in] mapsFolder A string containing the name for mapsFolder. No certainty is given about the form of this string
   * \param[in] _templatesFolder A String containing the name of the folder containing the number templates.
   * \param[in] _mapsNames A Tuple containing the names of the maps. These are not paths but just names.
   * \param[in] _mapsUnNames A Tuple containing the names of the undistorted maps. These are not paths but just names.
   * \param[in] _calibrationFile A string containing the name to the file containing the data for the calibration.
   * \param[in] _intrinsicCalibrationFile A string containing the name to the file containing the values of the matrix for the calibration.
   * \param[in] _blackMask Filter for black.
   * \param[in] _redMask Filter for red.
   * \param[in] _greenMask Filter for green.
   * \param[in] _victimMask Filter for the victims.
   * \param[in] _blueMask Filter for blue.
   * \param[in] _robotMask Filter for the triangle above the robot.
   * \param[in] _kernelSide
   * \param[in] _convexHullFile A String containing the name to file containing the points of the elements in the arena.
   * \param[in] _templates A Tuple containing the names of the templates. These are not paths but just names.
   */
	void save (
			string _baseFolder="data/",
			string _mapsFolder="map/",
			string _templatesFolder="num_template/",
			vector<string> _mapsNames={},
			vector<string> _mapsUnNames={},
			string _intrinsicCalibrationFile="intrinsic_calibration.xml",
			string _calibrationFile="calib_config.xml",
			Filter _blackMask=Filter(0, 0, 0, 179, 255, 70),
			Filter _redMask=Filter(15, 100, 140, 160, 255, 255),
			Filter _greenMask=Filter(54, 74, 25, 119, 255, 88),
			Filter _victimMask=Filter(0, 0, 0, 179, 255, 80),
			Filter _blueMask=Filter(100, 100, 40, 140, 200, 170),
			Filter _roboteMask=Filter(100, 100, 40, 140, 200, 170),
      		int _kernelSide=9,
			string _convexHullFile="convexHull.xml",
			vector<string> _templates={}
	);

	/*!\brief Function to write settings to file. Default is data/settings.xml.
	 *
	 * \param[in] _path Path to the file. Mind that it doesn't require the name of the file.
	 */
	void writeToFile(string _path="");

  /*! \brief Function to read from file. The data found is going to be added to the settings. Default file is data/settings.xml
   *
	 * \param[in] _path Path to the file. Mind that it doesn't require the name of the file.
   */
	void readFromFile(string _path="");

	/*! \brief Function to clean all settings: number types are set to 0, string are set to "", Tuples are set to Tuple<>() and Filter are set to all 0s.
	 *
	 */
	void clean();

	/*! \brief Function to clean all settings and then read from file. If no path is given the baseFolder is used.
	 * \param[in] _path Path to the file. Mind that it doesn't require the name of the file.
	 */
	void cleanAndRead(string _path="");

	/*!\brief Function to return the paths of maps. If ids are not specified all maps are returned.
	 *
	 * \param ids A Tuple containing the ids (that is the positions in this.mapsNames) of the maps to be retrieved.
	 * @return A Tuple containing the paths of the maps.
	 */
	Tuple<string> maps(Tuple<int> ids=Tuple<int>());

  /*!\brief Function to return the path of a map. If id is negative all maps are returned.
   *
   * \param id The positions in this.mapsNames of the map to be retrieved
   * @return A Tuple containing the paths of the maps.
   */
  Tuple<string> maps(int id=-1);

  /*!\brief A function to return the path of a given map.
   *
   * \param _mapName The name of the map to check in the Tuple.
   * @return The path to the map if the map is found, an empty string otherwise.
   */
  string maps(string _mapName);

  /*!\brief A function to return the paths of a given Tuple of maps.
   *
   * \param _mapNames A Tuple containing the names of the maps to check in the Tuple.
   * @return The paths to the maps if they are found, an empty Tuple otherwise.
   */
  Tuple<string> maps(Tuple<string> _mapNames);

  /**
   * @brief      Adds the name of an undistorted map to the list.
   *
   * @param[in]  unMap  The undistorted map
   *
   * @return     `true` of the name of the map could be added, `false` otherwise.
   */
  bool addUnMap(string unMap); 

  /*!\brief Function to return the paths of undistorted maps. If ids are not specified all undistorted maps are returned.
   *
   * \param ids A Tuple containing the ids (that is the positions in this.mapsUnNames) of the undistorted maps to be retrieved.
   * @return A Tuple containing the paths of the undistorted maps.
   */
	Tuple<string> unMaps(Tuple<int> ids=Tuple<int>());

  /*!\brief Function to return the path of an undistorted map. If id is negative all undistorted maps are returned.
   *
   * \param id The positions in this.mapsUnNames of the undistorted map to be retrieved
   * @return A Tuple containing the paths of the undistorted maps.
   */
  Tuple<string> unMaps(int id=-1);

  /*!\brief A function to return the path of a given undistorted map.
   *
   * \param _unMapName The name of the undistorted map to check in the Tuple.
   * @return The path to the undistorted map if it is found, an empty string otherwise.
   */
	string unMaps(string _unMapName);

  /*!\brief A function to return the paths of a given Tuple of undistorted maps.
   *
   * \param _unMapNames A Tuple containing the names of the undistorted maps to check in the Tuple.
   * @return The paths to the undistorted maps if they are found, an empty Tuple otherwise.
   */
	Tuple<string> unMaps(Tuple<string> _unMapNames);

	/*!\brief Function to return the path of a template. If id is negative all templates are returned.
   *
   * \param id The positions in this.templates of the template to be retrieved
   * @return A Tuple containing the paths of the templates.
   */
  Tuple<string> getTemplates(int id=-1);

	/*!\brief A function to return the path of a given template.
   *
   * \param _templateName The name of the template to check in the Tuple.
   * @return The path to the template if it is found, an empty string otherwise.
   */
	string getTemplates(string _template);

	/*!\brief A function to return the paths of a given Tuple of templates.
   *
   * \param _template A Tuple containing the names of the templates to check in the Tuple.
   * @return The paths to the templates if they are found, an empty Tuple otherwise.
   */
	Tuple<string> getTemplates(Tuple<string> _templates);

	enum COLOR {BLACK, RED, GREEN, VICTIMS, BLUE, ROBOT}; ///<Colors refered to the filters.

	/*!\brief Change the values of Tuple of filters. Mind that no write function is called.
	 *
	 * \param color A Tuple containing the colors of the filters to change.
	 * \param fil The new filters to be stored.
	 */
	void changeMask(Tuple<COLOR> color, Tuple<Filter> fil);

	/*!\brief Change the values of a filter. Mind that no write function is called.
   *
   * \param color The filter to change.
   * \param fil The new filter to be stored.
   */
	void changeMask(COLOR color, Filter fil);


	/*!\brief A function that creates a stringstream to print the values stored in settings.
	 *
	 * @return A strinstream containing the settings values.
	 */
	stringstream to_string () const {
		stringstream out;
		out << NAME(baseFolder) << ": " << baseFolder << endl;
		out << NAME(mapsNames) << ": " << mapsNames << endl;
		out << NAME(mapsUnNames) << ": " << mapsUnNames << endl;
		out << NAME(calibrationFile) << ": " << calibrationFile << endl;
		out << NAME(intrinsicCalibrationFile) << ": " << intrinsicCalibrationFile << endl;
		out << NAME(blackMask) << ": " << blackMask << endl;
		out << NAME(redMask) << ": " << redMask << endl;
		out << NAME(greenMask) << ": " << greenMask << endl;
		out << NAME(blueMask) << ": " << blueMask << endl;
    out << NAME(victimMask) << ": " << victimMask << endl;
    out << NAME(robotMask) << ": " << robotMask << endl;
		out << NAME(kernelSide) << ": " << kernelSide << endl;
		out << NAME(templatesFolder) << ": " << templatesFolder << endl;
		out << NAME(convexHullFile) << ": " << convexHullFile << endl;
		out << NAME(templates) << ": " << templates << endl;
		return out;
	}


	/*! This function overload the << operator so to print with `std::cout`.
			\param[in] out The out stream.
			\param[in] datThe settings to print.
			\returns An output stream to be printed.
	*/
	friend ostream& operator<<(ostream &out, const Settings& data) {
		out << data.to_string().str();
		return out;
	}

};

extern Settings *sett; ///<Global variable defined in main.cc.

#endif
