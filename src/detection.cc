#include "detection.hh"

vector<Mat> templates;

/*! \brief Loads some images and detects shapes according to different colors.

    \param[in] _imgRead Boolean flag that says if load or not the image from file or as a function parameter. True=load from file.
    \param[in] img The imgage that eventually is loaded from the function.
    \returns Return 0 if the function reach the end.
*/
int detection(const bool _imgRead, const Mat * img){
    load_number_template();

    for(int f=0; f<(_imgRead ? sett->mapsUnNames.size() : 1); f++){
        // Load image from file
        Mat un_img;
        if(_imgRead){
            string filename=sett->unMaps(f).get(0);
            cout << "Elaborating file: " << filename << endl;
            // Load unwrapped image from file
            un_img = imread(filename.c_str());
            if(un_img.empty()) {
                throw runtime_error("Failed to open the image " + filename);
            }
        } else{
            un_img = *img;
        }
        #ifdef WAIT
            my_imshow("unwrapped image", un_img, true);
        #endif

        //Convert from RGB to HSV
        Mat hsv_img;
        cvtColor(un_img, hsv_img, COLOR_BGR2HSV);

        //detection over the three values of the array
        COLOR_TYPE tmpVectColors[] = {RED, GREEN, BLUE, CYAN};
        for(int i=0; i<4; i++){
            shape_detection(hsv_img, tmpVectColors[i]);
            #ifdef WAIT
                mywaitkey();
                // if(i!=3){
                //     destroyAllWindows();
                // }
            #endif
        }
    }
    return(0);
}

/*! \brief The function simply store the value of the given matrix and allow the access to it from different function location. 
    \details The transformation matrix are computed in the unwrapping phase and taken from the localization.

    \param[in] transf It is the matrix that can be stored but also retrieved.
    \param[in] get It is the flag that says if the given matrix need to be stored or retrieved.
*/
void getConversionParameters(Mat & transf, const bool get){
    cout << "getConversionParameters\n";
    static Mat tr;
    if(get){
        transf = tr;
    } else{
        tr = transf;
        cout << "transformation matrix:\n" << tr << endl;
    }
}

/*! \brief Identify the loation of the robot by acquiring the image from the default camera of the environment.

    \returns The configuration of the robot in this exactly moment.
*/
Configuration2<double> localize(){
    //acquire the img and call the other localize function
    Mat img = acquireImage(false);

    return( localize(img, true) );
}

vector<Point> robotShape;
/*! \brief Identify the location of the robot respect to the given image.

    \param[in] img It is the image where the robot need to be located.
    \param[in] raw It is a boolean flag that says if the img is raw and need filters or not.
    \returns The configuration of the robot in this exactly moment, according to the image.
*/
Configuration2<double> localize(const Mat & img, const bool raw){
    cout << "localize0\n" << flush;

    static bool firstRun = true;
    static Mat transf, camera_matrix, dist_coeffs;
    if(firstRun){ //executed only at the first iteration of this function
        firstRun = false;
        const string calib_file = sett->intrinsicCalibrationFile;
        loadCoefficients(calib_file, camera_matrix, dist_coeffs);

        getConversionParameters(transf, true);
    }
    cout << "localize1\n" << flush;

    if(raw){
        cout << "RAW RAW RAW RAW RAW RAW RAW RAW RAW RAW RAW RAW \n";
        Mat fix_img;
        undistort(img, fix_img, camera_matrix, dist_coeffs);

        //Convert from RGB to HSV= Hue-Saturation-Value
        Mat hsv_img;
        cvtColor(fix_img, hsv_img, COLOR_BGR2HSV);
        #ifdef WAIT
            my_imshow("Img for localize", hsv_img, false);
            mywaitkey('q');
        #endif
        
        shape_detection(hsv_img, CYAN);//find robot
    } else{
        cout << "CLEAN CLEAN CLEAN CLEAN CLEAN CLEAN CLEAN CLEAN \n";
        shape_detection(img, CYAN);//find robot
    }
    cout << "localize2\n" << flush;

    
    //compute barycenter of the robot
    //the barycenter is the mean if the points are 3. Otherwise we also compute the mean over x and y.
    if(robotShape.size()!=3){
        cout << "Warning: The robot is not well defined (not 3 points found but " << robotShape.size() << ").\n\n";
    }
    cout << "robotShape size " << robotShape.size() << endl;
    cout << "From localize:\n";
    vector<Point2f> vpOut;
    for(Point p : robotShape){
        cout << p.x << " - " << p.y << endl;  
    }
    cout << "localize A\n" << flush;
    // apply conversion to the right reference system
    // https://stackoverflow.com/questions/30194211/opencv-applying-affine-transform-to-single-points-rather-than-entire-image
    // 
    vector<Point2f> convert(1);
    for(Point p : robotShape){
        convert[0] = p;
        perspectiveTransform(convert, convert, transf);  //maybe a simple matrix multiplication will be faster...    }
        cout << "Trasforming: " << p << " to: " << convert[0] << endl;
        vpOut.push_back(convert[0]);
    }
    cout << "vpOut size: " << vpOut.size() << endl;
    /*/
    for(Point p : robotShape) 
        vpOut.push_back(Point2f((float)p.x, (float)p.y));
    //*/
    cout << "localize B\n" << flush;
    double xAvg=0, yAvg=0;
    for(Point p : vpOut){
        xAvg += p.x;
        yAvg += p.y;
    }
    xAvg /= vpOut.size()*1.0;
    yAvg /= vpOut.size()*1.0;

    cout << "localize C\n" << flush;
    Point2<double> confPoint(xAvg, yAvg);
    cout << "Barycenter (AKA centroid): " << confPoint << endl;

    cout << "localize D\n" << flush;
    double Dist=0;
    Point2<int> tail;
    for (Point p : vpOut){
        Point2<int> app = Point2<int>(p);
        double dist=app.distance(confPoint);
        if (dist>Dist){
            tail=app;
            Dist=dist;
        }
    }

    cout << "localize E\n" << flush;
    Configuration2<double> conf(confPoint, tail.th(confPoint));
    cout << "tail of the robot: " << tail << endl;
    cout << "New robot position:     " << conf.point() << ", " << conf.angle().toDeg() << "° " << conf.angle().toRad()/3.14 << "pi" << endl;
    cout << "localize F\n" << flush;
    mywaitkey('q');
    cout << "localize G\n" << flush;
    return(conf);
}



/*! \brief Load some templates and save them in the global variable 'templates'.
*/
void load_number_template(){ //load the template for number recognition
    Mat tmp;
    for(auto el : sett->getTemplates(-1)){
        tmp = imread(el);
        cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
        bitwise_not(tmp, tmp);
        templates.push_back(tmp);
    }
}

/*! \brief Detect shapes inside the image according to the variable 'color'.

    \param[in] img Image on which the research will done.
    \param[in] color It is the type of reference color. These color identify the possible spectrum that the function search on the image.
*/
void shape_detection(const Mat & img, const COLOR_TYPE color){
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    Filter mask;
    switch(color){
        case RED: {
            cout << "\tObstacles detection\n";  
            mask = sett->redMask;
            break;
        }
        case GREEN: {  
            cout << "\tVictim detection\n";   
            mask = sett->greenMask; 
            break;
        }
        case BLUE: {
            cout << "\tGate detection\n";    
            mask = sett->blueMask;   
            break;
        }
        case CYAN: {
            cout << "\tRobot localization\n";    
            mask = sett->robotMask;   
            break;
        }
        default:
            break;
    }
    
    Mat color_mask;
    inRange(img, mask.Low(), mask.High(), color_mask);
    
    #ifdef DEBUG
        my_imshow("Color_filter", color_mask);
    #endif

    find_contours(color_mask, img, color);
}

/*! \brief It apply some filtering function for isolate the subject and remove the noise.
    \details An example of the sub functions called are: GaussianBlur, Erosion, Dilation and Threshold.

    \param[in, out] img Is the image on which the function apply the filtering.
    \param[in] color It is the type of reference color. According to the color the filtering functions apply can change in the type and in the order.
*/
void erode_dilation(Mat & img, const COLOR_TYPE color){
    // It is now called only for color=BLACK...
    const int erode_side = sett->kernelSide; //odd number
    const int center = erode_side/2+1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(erode_side, erode_side), Point(center, center) );
    // 0red && 2blue    -> smooth - erode - dilation - smooth - treshold
    // 1green && 3black -> smooth - dilation - smooth - erode - treshold

    //smooth -> gaussian blur
    GaussianBlur(img, img, Size(erode_side, erode_side), 1, 1);

    if(color==RED || color==BLUE || color==BLACK){
        // Apply the erode operation
        erode(img, img, kernel);
    }

    // Apply the dilation operation
    dilate(img, img, kernel);

    //smooth -> gaussian blur
    GaussianBlur(img, img, Size(erode_side, erode_side), 1, 1);

    if(color==GREEN){
        // Apply the erode operation
        erode(img, img, kernel);
    }

    threshold(img, img, 254, 255, 0 ); // threshold and binarize the image, to suppress some noise
}

bool _compare (  const pair<int, int > & a, 
                const pair<int, int > & b ){ 
    return (a.first<b.first); 
}

/*! \brief Given an image, in black/white format, identify all the borders that delimit the shapes.

    \param[in] img It is an image in HSV format at the base of the elaboration process.
    \param[out] original It is the original source of 'img', it is used for showing the detected contours.
    \param[in] color It is the type of reference color.
*/
#define EPS_CURVE 5
void find_contours( const Mat & img,
                    const Mat & original, 
                    const COLOR_TYPE color)
{
    #define MIN_AREA_SIZE 2000 //defined as pixels^2 (in our scenaria it means mm^2)
    vector<vector<Point>> contours, contours_approx;
    vector<Point> approx_curve;
    vector<int> victimNum;
    
    // The function erode_dilation is not called (but eventually this is the right place)...
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // find external contours of each blob

    if(color==GREEN){cout << "\tNumber detection\n";}
    for (vector<Point> contour : contours){
        if (contourArea(contour) > MIN_AREA_SIZE){ // filter too small contours to remove false positives
            approxPolyDP(contour, approx_curve, EPS_CURVE, true); // fit a closed polygon (with less vertices) to the given contour, with an approximation accuracy (i.e. maximum distance between the original and the approximated curve) of EPS_CURVE (=5)
            
            if(color==GREEN){
                Rect blob = boundingRect(Mat(approx_curve)); // find bounding box for each green blob
                int num_detect = number_recognition(blob, original);
                if(num_detect!=-1){
                    contours_approx.push_back(approx_curve);
                    victimNum.push_back(num_detect);
                }
            } else{
                contours_approx.push_back(approx_curve);
            }
        }
    }
    #ifdef WAIT
        Scalar s;
        switch(color){
            case RED:   s = Scalar(0,0,255);    break;
            case BLUE:  s = Scalar(255,0,0);    break;
            case BLACK: s = Scalar(5,5,5);      break;
            case GREEN: s = Scalar(0,255,0);    break;
            case CYAN:  s = Scalar(255,255,0);  break;
            default:    s = Scalar(0,170,220);  break;
        }
        drawContours(original, contours_approx, -1, s, 5, LINE_AA);
        my_imshow("Detected shape", original);
    #endif

    if(color==CYAN){
        // the points are returned thanks to a global variable.
        if(contours_approx.size()==1){
            robotShape = contours_approx[0];
        } else{
            cout << "Warning: Not well defined robot filter.\n\tThere are " << contours_approx.size() << " possible blobs.\n" << flush;
            cout << "The one with the biggest area will be choosen\n\n" << flush;
            double area, minArea = contourArea(contours_approx[0]);
            int maxIndex = 0;
            for(unsigned int i=1; i<contours_approx.size(); i++){
                area = contourArea(contours_approx[i]);
                if(area>minArea){
                    minArea = area;
                    maxIndex = i;
                }
            }
            robotShape = contours_approx[maxIndex];
        }
    } else{
        // sort the victims' vector of points according to their numbers.
        if(color==GREEN){
            vector<pair<int, int > > vicPoints;
            for (uint i=0; i<victimNum.size(); i++){
                vicPoints.push_back(make_pair(victimNum[i], i));
            }
            std::sort(vicPoints.begin(), vicPoints.end(), _compare);
            vector<vector<Point> > tmp;
            for (auto el : vicPoints){
                tmp.push_back(contours_approx[el.second]);
            }
            contours_approx.swap(tmp);
        }

        save_convex_hull(contours_approx, color);
    }
}

/*! \brief Given some vector save it in a xml file.

    \param[in] contours Is a vector that is saved in a xml file.
    \param[in] color It is the type of reference color, according to which the function decide if saved ('color==GREEN') or not ('otherwise') the vector 'victims'.
*/
void save_convex_hull(  const vector<vector<Point>> & contours,
                        const COLOR_TYPE color)
{
    vector<vector<Point>> hull;
    vector<Point> hull_i;
    for(unsigned i=0; i<contours.size(); i++){
        convexHull(contours[i], hull_i, true);//return point in clockwise order
        hull.push_back(hull_i);
    }
    string save_file = sett->convexHullFile;
    static FileStorage fs(save_file, FileStorage::WRITE);

    string str;
    switch(color){
        case RED:   {str="obstacles"; break;}
        case GREEN: {str="victims"; break;}
        case BLUE:  {str="gate"; break;}
        default:    break;
    }

    fs << str << hull;
    
    if (color==BLUE){
        fs.release();
    }
}

/*! \brief Detect a number on an image inside a region of interest.

    \param[in] blob Identify the region of interest inside the image 'base'.
    \param[in] base Is the image where the function will going to search the number.

    \returns The number recognise, '-1' otherwise.
*/
int number_recognition(Rect blob, const Mat & base){ //filtering
    Mat processROI(base, blob); // extract the ROI containing the digit
    if(processROI.empty()){return(-1);}
    
    resize(processROI, processROI, Size(200, 200)); // resize the ROI

    // black filter
    #ifdef DEBUG
        my_imshow("before black filter", processROI, true);
    #endif
    Filter mask = sett->victimMask;
    inRange(processROI, mask.Low(), mask.High(), processROI);
    #ifdef WAIT
        my_imshow("before erode", processROI);
    #endif

    erode_dilation(processROI, BLACK);
    #ifdef WAIT
        my_imshow("ROI filtered", processROI);
    #endif
    // crop out the number if it is possible
    crop_number_section(processROI);

    // matching template
    // Find the template digit with the best matching
    double maxScore = 1e7;  // I don't know what this number represents...
    int maxIdx = -1;
    for (unsigned i=0; i<templates.size(); i++) {
        Mat result;
        Mat _template; 
        resize(templates[i], _template, Size(200, 200));
        resize(processROI, processROI, Size(200, 200));
        matchTemplate(processROI, _template, result, TM_CCOEFF); //TM_SQDIFF

        double score;
        minMaxLoc(result, nullptr, &score);
        if (score > maxScore) {
            maxScore = score;
            maxIdx = i;
        }
    }
    #ifdef DEBUG
        cout << "Best fitting template: -> " << maxIdx << "->" << maxIdx%10 << " <- with score of: " << maxScore << endl << endl;
        mywaitkey();
    #endif
    return(maxIdx%10);  //if we have 20-30-... templates it return the true number
}

/*! \brief Given an image identify the region of interest(ROI) and crop it out. 

    \param[in,out] ROI Is the image that the function will going to elaborate.
*/
void crop_number_section(Mat & ROI){
    // Tutorial for the min rectangle arround a shape. 
    // https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rotated_ellipses/bounding_rotated_ellipses.html
    vector<vector<Point>> contours;
    vector<Point> contour;

    findContours(ROI, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if(contours.size()==1){
        contour = contours[0];
    } else{
        //use the convex hull of all points instead of a group of different points
        vector<Point> tmpContour;
        for(unsigned i=0; i<contours.size(); i++){
            for(unsigned j=0; j<contours[i].size(); j++){
                tmpContour.push_back(contours[i][j]);
            }
        }
        if(tmpContour.size()==0){
            return; // extreme rare case that can occours if the filters are not well setted (it cause core dumped)
        }
        convexHull(tmpContour, contour, true);//return points in clockwise order
    }
    
    // Find the rotated rectangles for the contour
    RotatedRect minRect = minAreaRect(contour);

    // rotated rectangle
    Point2f rect_points[4]; 
    minRect.points( rect_points );

    //alias for semplicity
    float x = rect_points[0].x;
    float y = rect_points[0].y;
    float w = minRect.size.width;
    float h = minRect.size.height;

    #ifdef DEBUG
        // Draw contour + rotated rect
        Mat drawing = Mat::zeros( ROI.size(), CV_8UC3 );
        RNG rng(12345);
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // contour
        contours.resize(0);
        contours.push_back(contour);
        drawContours(drawing, contours, -1, color, 1, 8, vector<Vec4i>(), 0, Point() );
        cout << endl;
        for( int i = 0; i < 4; i++ ){
            //cout << "point[" << i << "] x: " << rect_points[i].x << " y: " << rect_points[i].y << endl;
            line( drawing, rect_points[i], rect_points[(i+1)%4], color, 1, 8 );
        }
        cout << "angle: " << minRect.angle << " x: " << x << " y: " << y << " width: " << w  << " height: " << h << endl;
        // Show in a window
        my_imshow("Contour", drawing );
    #endif

    // How RotatedRect angle work: https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
    Mat corner_pixels, transf_pixels;
    Size size;
    if(h > w){
        // the orientation of the rectangle is to the left
        corner_pixels = (Mat_<float>(4,2) << rect_points[1].x, rect_points[1].y, rect_points[2].x, rect_points[2].y, rect_points[3].x, rect_points[3].y, x, y);
        transf_pixels = (Mat_<float>(4,2) << 0, 0, w, 0, w, h, 0, h);
        size = Size(w, h);
    } else{
        // the orientation of the rectangle is to the right
        corner_pixels = (Mat_<float>(4,2) << rect_points[2].x, rect_points[2].y, rect_points[3].x, rect_points[3].y, x, y, rect_points[1].x, rect_points[1].y);
        transf_pixels = (Mat_<float>(4,2) << 0, 0, h, 0, h, w, 0, w);
        size = Size(h, w);
    }        

    Mat transf = getPerspectiveTransform(corner_pixels, transf_pixels);

    Mat rotNumber;
    warpPerspective(ROI, ROI, transf, size);

    #ifdef DEBUG
        my_imshow("rotated Num", ROI);
    #endif

}