/*! 
	\file calibration.hh  
	\brief Library for calibration
*/
#ifndef CALIBRATION_HH
#define CALIBRATION_HH

#include "utils.hh"

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

class Settings{
	public:
	    Settings() : goodInput(false) {}	///< Constructor that sets `goodInput` to false
			enum Pattern {NOT_EXISTING=0, CHESSBOARD=1};
			enum InputType {INVALID=0, IMAGE_LIST=3};

	    void write(FileStorage& fs) const;	
	    void read(const FileNode& node);	
	    void validate();					
	    Mat nextImage();					
	    static bool readStringList( const string& filename, vector<string>& l ); 
	    static bool isListOfImages( const string& filename);					 

	public:
	    Size boardSize;              ///< The size of the board -> Number of items by width and height
	    Pattern calibrationPattern = CHESSBOARD; ///< One of the Chessboard, circles, or asymmetric circle pattern
	    float squareSize;            ///< The size of a square in your defined unit (point, millimeter,etc).
	    int nrFrames;                ///< The number of frames to use from the input for calibration
	    float aspectRatio;           ///< The aspect ratio
	    int delay;                   ///< In case of a video input
	    bool writePoints;            ///< Write detected feature points
	    bool writeExtrinsics;        ///< Write extrinsic parameters
	    bool calibZeroTangentDist;   ///< Assume zero tangential distortion
	    bool calibFixPrincipalPoint; ///< Fix the principal point at the center
	    bool flipVertical;           ///< Flip the captured images around the horizontal axis
	    string outputFileName;       ///< The name of the file where to write
	    bool showUndistorsed;        ///< Show undistorted images after calibration
	    string input;                ///< The input 
	    bool useFisheye = false;     ///< use fisheye camera model for calibration
	    bool fixK1;                  ///< fix K1 distortion coefficient
	    bool fixK2;                  ///< fix K2 distortion coefficient
	    bool fixK3;                  ///< fix K3 distortion coefficient
	    bool fixK4;                  ///< fix K4 distortion coefficient
	    bool fixK5;                  ///< fix K5 distortion coefficient

	    int cameraID;
	    vector<string> imageList;
	    size_t atImageList;
	    VideoCapture inputCapture;
	    InputType inputType = IMAGE_LIST;
	    bool goodInput;
	    int flag;

	private:
	    string patternToUse;
};


int calibration( 
			const string inputFile="data/calib_config.xml"
		);

bool runCalibrationAndSave( Settings& s, 
							Size imageSize, 
							Mat&  cameraMatrix, 
							Mat& distCoeffs,
                           	vector<vector<Point2f> > imagePoints 
						);

#endif