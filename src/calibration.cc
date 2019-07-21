#include"calibration.hh"

/*! \brief Write serialization.
    \details This function write data to a file.
    
    \param[in] fs The filename where to write.
*/
void CalSettings::write(FileStorage& fs) const 
{
    fs << "{"
              << "BoardSize_Width"  << boardSize.width
              << "BoardSize_Height" << boardSize.height
              << "Square_Size"         << squareSize
              << "Calibrate_Pattern" << patternToUse
              << "Calibrate_NrOfFrameToUse" << nrFrames
              << "Calibrate_FixAspectRatio" << aspectRatio
              << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
              << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint
              << "Write_DetectedFeaturePoints" << writePoints
              << "Write_extrinsicParameters"   << writeExtrinsics
              << "Write_outputFileName"  << outputFileName
              << "Show_UndistortedImage" << showUndistorsed
              << "Input_FlipAroundHorizontalAxis" << flipVertical
              << "Input_Delay" << delay
              << "Input" << input
       << "}";
}

/*! \brief Read serialization.
    \details This function read data from a file and stores each node in their corresponding variables.
    
    \param[in] node The node of the file to consider.
*/
void CalSettings::read(const FileNode& node) 
{
    node["BoardSize_Width" ] >> boardSize.width;// >> boardSize.width;
    node["BoardSize_Height"] >> boardSize.height;
    node["Calibrate_Pattern"] >> patternToUse;
    node["Square_Size"]  >> squareSize;
    node["Calibrate_NrOfFrameToUse"] >> nrFrames;
    node["Calibrate_FixAspectRatio"] >> aspectRatio;
    node["Write_DetectedFeaturePoints"] >> writePoints;
    node["Write_extrinsicParameters"] >> writeExtrinsics;
    node["Write_outputFileName"] >> outputFileName;
    node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
    node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
    node["Calibrate_UseFisheyeModel"] >> useFisheye;
    node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
    node["Show_UndistortedImage"] >> showUndistorsed;
    node["Input"] >> input;
    node["Input_Delay"] >> delay;
    node["Fix_K1"] >> fixK1;
    node["Fix_K2"] >> fixK2;
    node["Fix_K3"] >> fixK3;
    node["Fix_K4"] >> fixK4;
    node["Fix_K5"] >> fixK5;
  
    validate();
}

/*! \brief This function validate the content of the file. 
    \details Even though this function doesn't return anything nor has any parameters 
    for output, it sets a variable of the `CalSettings` class, that is `googInput`, 
    to `false` if some infos were wrong. `true` otherwise.
*/
void CalSettings::validate()
{ 
    goodInput = true;
    ///The options it takes in consideration are the following:
    ///* Size must be positive.\n
    if (boardSize.width <= 0 || boardSize.height <= 0){
        cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
        goodInput = false;
    }
    ///* Cells must be greater than \f$10^{-6}\f$.
    if (squareSize <= 10e-6){
        cerr << "Invalid square size " << squareSize << endl;
        goodInput = false;
    }
    ///* The number of frames considered, that is images, must be greater than 0.
    if (nrFrames <= 0){
        cerr << "Invalid number of frames " << nrFrames << endl;
        goodInput = false;
    }

    ///* Check for valid input, that is a valid list of images.
    if (input.empty()) {
        inputType = INVALID;
    }
    ///* Else a list of image is being used.
    else{
        if (isListOfImages(input) && readStringList(input, imageList)){
            inputType = IMAGE_LIST;
            nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
        }
        else {
            inputType = INVALID;
        }
    }
    if (inputType == INVALID){
        cerr << " Input does not exist: " << input;
        goodInput = false;
    }

    flag = 0;
    if(calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
    if(calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
    if(aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
    if(fixK1)                  flag |= CALIB_FIX_K1;
    if(fixK2)                  flag |= CALIB_FIX_K2;
    if(fixK3)                  flag |= CALIB_FIX_K3;
    if(fixK4)                  flag |= CALIB_FIX_K4;
    if(fixK5)                  flag |= CALIB_FIX_K5;

    ///* Check the field pattern: if it doesn't correspond to a known one than it's invalid.
    if (!patternToUse.compare("CHESSBOARD")) {
        calibrationPattern = CHESSBOARD;
    }
    else {
        calibrationPattern = NOT_EXISTING;
        cerr << " Camera calibration mode does not exist: " << patternToUse << endl;
        goodInput = false;
    }
    atImageList = 0;

}

/*! \brief Get next image from list.
    \returns A matrix containing the next image to consider.
*/
Mat CalSettings::nextImage()
{
    return imread(imageList[atImageList++], IMREAD_COLOR);
}

/*! \brief Read from file a list of images.
    
    \param[in] filename The name of the file from which to read.
    \param[out] l A vector which will contain the names of the file from the list.

    \return `false` if the file could not be opened or if the file doesn't contain a list\n `true` otherwise.
*/
bool CalSettings::readStringList(  const string& filename, 
                                vector<string>& l )
{
    l.clear();
    //Open file for reading 
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() ) {
        return false;
    }
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ ) {
        return false;
    }
    //Read all nodes in file and insert them in `l`.
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it ){
        l.push_back((string)*it);
    }
    return true;
}

/*! \brief Check if the file from which is trying to retrive a list is a valid format (xml or yaml).
    \param[in] filename The name of the file to check for validity.
    \return `false` is the file is not xml or yaml\n `true` otherwise.
*/
bool CalSettings::isListOfImages( const string& filename)
{
    string s(filename);
    // Look for file extension
    if( s.find(".xml") == string::npos && s.find(".yaml") == string::npos && s.find(".yml") == string::npos ){
        return false;
    }
    else {
        return true;
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*! 
    \brief Function to run the complete calibration. 

    \param[in] inputFile Name of the setting.xml file. It's set to default to default.xml

    \return -2 if the CalSettings file could be load but the input was not well-formed\n
            -1 if the CalSettings file could not be opened.\n
            0 if everything went fine.
*/
int calibration(string inputFile)
{
    Settings *set=new Settings();
    set->readFromFile();

    inputFile=set->calibrationFile;
    inputFile=(inputFile=="" ? set->calibrationFile : inputFile);
    //file_read
    CalSettings s;

    const string inputCalSettingsFile = inputFile;

    FileStorage fs(inputCalSettingsFile, FileStorage::READ); //open the file for reading
    if (!fs.isOpened()){
        cerr << "Could not open the configuration file: \"" << inputCalSettingsFile << "\"" << endl;
        return -1;
    }
    
    fs["Settings"] >> s; //read everything from file in node "Settings"
    fs.release(); // close Settings file

    //FileStorage fout("CalSettings.yml", FileStorage::WRITE); // write config as YAML
    //fout << "CalSettings" << s;

    if (!s.goodInput)
    {
        cerr << "Invalid input detected. Application stopping. " << endl;
        return -2;
    }

    //Define variables 
    vector<vector<Point2f> > imagePoints; //Well... this is quite a nice question
    Mat cameraMatrix, distCoeffs; //Matrixes for camera parameters and distortion coefficients
    Size imageSize; //lenght and width of image
    int mode = CAPTURING; //Set the mode based if there is a list of images or not
    const Scalar RED(0,0,255), GREEN(0,255,0);
    #ifdef DEBUG
        const char ESC_KEY = 27 ;
    #endif
    //get_input

    for(;;)
    {
        Mat view; //The matrix in which the image being considered is stored
        bool blinkOutput = false;
        //Store next image
        view = s.nextImage(); 

        // If no more image, or got enough, then stop calibration and show result
        if( mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames )
        {
            if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
              mode = CALIBRATED;
            else
              mode = DETECTION;
        }
        // If there are no more images stop the loop
        if(view.empty())
        {   
            // if calibration threshold was not reached yet, calibrate now
          if( mode != CALIBRATED && !imagePoints.empty() )
            runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
          break;
        }

        //get_input
        imageSize = view.size();  // Format input image.
        if( s.flipVertical )    {
            flip( view, view, 0 );
        }

        //find_pattern
        vector<Point2f> pointBuf;

        bool found;

        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

        found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
        
        //find_pattern
        if ( found)                // If done with success,
        {
              // improve the found corners' coordinate accuracy for chessboard
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(   viewGray, pointBuf, Size(11,11), Size(-1,-1), 
                            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));

          if( mode == CAPTURING) {
              imagePoints.push_back(pointBuf);
          }
            // Draw the corners.
            drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );
        }
        //pattern_found
        //----------------------------- Output Text ------------------------------------------------
        //output_text
        string msg = (mode == CAPTURING) ? "100/100" :
                      mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
        }

        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);

        if( blinkOutput )
            bitwise_not(view, view);
        //output_text
        //------------------------- Video capture  output  undistorted ------------------------------
        //output_undistorted
        if( mode == CALIBRATED && s.showUndistorsed )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }
        //output_undistorted
        //------------------------------ Show image and check for input commands -------------------
        //await_input

        #ifdef DEBUG
            imshow("Image View", view);
        
            char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);
            if( key  == ESC_KEY )
                break;

            if( key == 'u' && mode == CALIBRATED )
               s.showUndistorsed = !s.showUndistorsed;

            if( s.inputCapture.isOpened() && key == 'g' )
            {
                mode = CAPTURING;
                imagePoints.clear();
            }
        #endif
        //await_input
    }
    //End of for cycle

    // -----------------------Show the undistorted image for the image list ------------------------
    //show_results
    if( s.inputType == CalSettings::IMAGE_LIST && s.showUndistorsed )
    {
        Mat view, rview, map1, map2;
//        Mat optimalCamera = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
        initUndistortRectifyMap(
            cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);

        for(size_t i = 0; i < s.imageList.size(); i++ )
        {
            view = imread(s.imageList[i], IMREAD_COLOR);
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
            #ifdef DEBUG
                imshow("Image View", rview);
                char c = (char)waitKey();
                if( c  == ESC_KEY || c == 'q' || c == 'Q' )
                    break;
            #endif
        }
    }
    //show_results
	
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*! \brief Reads CalSettings from file. If there is none then initiate a new `CalSettings`.
    
    \param[in] node: node to consider for getting CalSettings;
    \param[in] x: `CalSettings` to configure;
    \param[in] default_value: `CalSettings` default value. Setted to `CalSettings()`.
*/
static inline void read(const FileNode& node, 
                        CalSettings& x, 
                        const CalSettings& default_value)
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

/*! \brief Compute the errors of the projection. 
    \param[in]  objectPoints The real image points which will be projected
    \param[in] rvecs Input vector of rotation vectors estimated for each pattern view.
    \param[in] tvecs Input vector of translation vectors estimated for each pattern view.
    \param[in] cameraMatrix The matrix containing the parameters for the camera
    \param[in] distCoeffs The matrix containing the distortion coefficients.
    \param[in] fisheye A variable which says if a fish eye correction should be applied or no.
    \param[out] perViewErrors A vector containing the error for each image.
    \param[out] imagePoints The projected points for each image.
    
    \returns The total error.
*/
static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, 
                                         const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix, 
                                         const Mat& distCoeffs,
                                         vector<float>& perViewErrors, 
                                         bool fisheye)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size()); //Vector that'll contain errors for each image

    for(size_t i = 0; i < objectPoints.size(); ++i ) {
        
        //This function projects points toward a plane. It takes the points from objectPoints and write the output vector imagePoints2.
        projectPoints(  objectPoints[i], rvecs[i], tvecs[i], 
                        cameraMatrix, distCoeffs, imagePoints2);

        err = norm(imagePoints[i], imagePoints2, NORM_L2); //Calculate an relative difference norm

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n); //compute err for picture
        totalErr        += err*err; 
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints); //compute general error
}

/*! \brief This function compute the position of the upper corners of every cell. 

    \param[in] boardSiz The dimension of the chess board.
    \param[in] squareSize The dimension of the edge of a cell.
    \param[out] corners A vector of Point3fs which equals to the corners of the cells. 
*/
void calcBoardCornerPositions(  Size boardSize, 
                                float squareSize, 
                                vector<Point3f>& corners)
{
    corners.clear();

    for( int i = 0; i < boardSize.height; ++i ) {
        for( int j = 0; j < boardSize.width; ++j ) {
            corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
            INFO((to_string(i)+", "+to_string(j)+"   ("+to_string((int)(j*squareSize))+", "+to_string((int)(i*squareSize))+")").c_str());
        }
    }
}

/*! \brief This function run the calibration creating the matrixed for the camera and the distorsion coefficients.

    \param[in] s The `CalSettings` read from the file and memorized.
    \param[in] imageSize The size of the image used in `calibrateCamera()` to initialize the camera matrix.
    \param[in] imagePoints The projected points for each image.
    \param[in] reprojErrs The re-projection error, that is a geometric error corresponding to the image distance between a projected point and a measured one. 
    \param[out] cameraMatrix The matrix of the camera parameters
    \param[out] distCoeffs The matrix of the distorsion coefficients. 
    \param[out] rvecs Output vector of rotation vectors estimated for each pattern view.
    \param[out] tvecs Output vector of translation vectors estimated for each pattern view.
    \param[out] totalAvgErr The total avarage error given from distorsion. 

    \returns `false` if one or more elements in the `cameraMatrix` and `distCoeffs` are invalid.\n `true` if all the elements are valid.
*/
static bool runCalibration( CalSettings& s, 
                            Size& imageSize, 
                            Mat& cameraMatrix, 
                            Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, 
                            vector<Mat>& rvecs, 
                            vector<Mat>& tvecs,
                            vector<float>& reprojErrs, 
                            double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( s.flag & CALIB_FIX_ASPECT_RATIO ) {
        cameraMatrix.at<double>(0,0) = s.aspectRatio;
    }
    
    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0]);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms;

    rms = calibrateCamera(  objectPoints, imagePoints, imageSize, 
                            cameraMatrix, distCoeffs, rvecs, 
                            tvecs, s.flag
                        );

    cerr << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, 
                                            tvecs, cameraMatrix, distCoeffs, 
                                            reprojErrs, s.useFisheye);

    return ok;
}

/*! \brief Function to save the computed parameters to a file. 
    \param[in] s Use the `CalSettings` got at the beginning for information as the output file name, image and board size. 
    \param[in] imageSize The size of the imgage.
    \param[in] cameraMatrix The camera matrix.
    \param[in] distCoeffs The distorsion coefficient matrix. 
    \param[int] rvecs Vector of rotation vectors estimated for each pattern view.
    \param[in] tvecs Vector of translation vectors estimated for each pattern view.
    \param[in] reprojErrs The re-projection error, that is a geometric error corresponding to the image distance between a projected point and a measured one. 
    \param[in] imagePoints The projected points for each image.
    \param[in] totalAvgErr The total avarage error given from distorsion.
*/
static void saveCameraParams(   const CalSettings& s, 
                                const Size& imageSize,
                                const Mat& cameraMatrix,
                                const Mat& distCoeffs,
                                const vector<Mat>& rvecs,
                                const vector<Mat>& tvecs,
                                const vector<float>& reprojErrs,
                                const vector<vector<Point2f> >& imagePoints,
                                const double totalAvgErr )
{
    /// Open file for writing
    FileStorage fs( s.outputFileName, FileStorage::WRITE );

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf), "%c", t2 );
    /// Stores time of calibration
    fs << "calibration_time" << buf;

    ///Store infos about the images
    if( !rvecs.empty() || !reprojErrs.empty() ) {
        fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
    }
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << s.boardSize.width;
    fs << "board_height" << s.boardSize.height;
    fs << "square_size" << s.squareSize;

    if( s.flag & CALIB_FIX_ASPECT_RATIO )
        fs << "fix_aspect_ratio" << s.aspectRatio;

    if (s.flag)
    {
        std::stringstream flagsStringStream;
        
        flagsStringStream << "flags:"
            << (s.flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
            << (s.flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
            << (s.flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
            << (s.flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
            << (s.flag & CALIB_FIX_K1 ? " +fix_k1" : "")
            << (s.flag & CALIB_FIX_K2 ? " +fix_k2" : "")
            << (s.flag & CALIB_FIX_K3 ? " +fix_k3" : "")
            << (s.flag & CALIB_FIX_K4 ? " +fix_k4" : "")
            << (s.flag & CALIB_FIX_K5 ? " +fix_k5" : "");

        fs.writeComment(flagsStringStream.str());
    }

    fs << "flags" << s.flag;

    fs << "fisheye_model" << s.useFisheye;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (s.writeExtrinsics && !reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if(s.writeExtrinsics && !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
        bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
        bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

        for( size_t i = 0; i < rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(int(i), int(i+1)), Range(0,3));
            Mat t = bigmat(Range(int(i), int(i+1)), Range(3,6));

            if(needReshapeR)
                rvecs[i].reshape(1, 1).copyTo(r);
            else
            {
                //*.t() is MatExpr (not Mat) so we can use assignment operator
                CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
                r = rvecs[i].t();
            }

            if(needReshapeT)
                tvecs[i].reshape(1, 1).copyTo(t);
            else
            {
                CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
                t = tvecs[i].t();
            }
        }
        fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
        fs << "extrinsic_parameters" << bigmat;
    }

    if(s.writePoints && !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( size_t i = 0; i < imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
}

/*! \brief Reads CalSettings from file. If there is none then initiate a new `CalSettings`.
    \param[in] s The `CalSettings` being used during the execution.
    \param[in] imageSize The dimensions of the images.
    \param[in] imagePoints The projected points for each image.
    \param[out] cameraMatrix The matrix which is used to store the values for the camera parameters.
    \param[out] distCoeffs The matrix which is used to store the distortion coefficients.

    \return `true` if the calibration succeded.\n `false` otherwise.
*/
bool runCalibrationAndSave( CalSettings& s, 
                            Size imageSize, 
                            Mat& cameraMatrix, 
                            Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, 
                            imagePoints, rvecs, tvecs, 
                            reprojErrs, totalAvgErr
                        );

    cerr << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re projection error = " << totalAvgErr << endl;

    if (ok) {
        saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, 
                        rvecs, tvecs, reprojErrs, imagePoints,
                        totalAvgErr
                    );
    }
    return ok;
}
