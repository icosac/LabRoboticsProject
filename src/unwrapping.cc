#include"unwrapping.hh"

#define AREA_RATIO 0.7
#define AREA_MIN 500 //pixel^2

static double distance(Point c1, Point c2);

/*! \brief Take some images according to a xml and unwrap the black rectangle inside the image after appling undistortion trasformation.
    \details Load from the xml file 'data/settings.xml' the name of some images, load the images from the file,\n
    apply the calibration (undistortion trasformation) thanks to the matrices load with the 'loadCoefficients' function.\n
    Then, with the use of a filter for the black the region of interest (a rectangle) is identified and all the perspective is rotated for reach a top view of the rectangle.\n
    Finally, the images are saved on some files.

    \param[in] _imgRead Boolean flag that says if load or not the image from file, or as a function parameter. In addition, also the return procedure change if true the image is saved on the disk otherwise is saved on the img function parameter. True=load and store on file.
    \param[in/out] img The image that eventually is loaded from the function. And the one that will be modified for returning the elaborated frame.
    \returns A 0 is return if the function reach the end.
*/
int unwrapping(const bool _imgRead, Mat * img){

    const string calib_file = sett->baseFolder+sett->intrinsicCalibrationFile;
    for(int f=0; f<(_imgRead ? sett->mapsNames.size() : 1); f++){
        // Load image from file
        Mat or_img;
        if(_imgRead){
            string filename = sett->maps(f).get(0);
            cout << "Elaborating file: " << filename << endl;
            or_img = imread(filename.c_str());
        } else{
            or_img = *img;
        }
        
        #ifdef UNDISTORT
            COUT(calib_file)
            cout << "loadCoefficients" << endl;
            // fix calibration with matrix
            Mat camera_matrix, dist_coeffs;
            loadCoefficients(calib_file, camera_matrix, dist_coeffs);

            cout << "loadCoefficients done" << endl;

            COUT(camera_matrix)
            COUT(dist_coeffs)
            // Display origina image
            #ifdef DEBUG
                my_imshow("Original", or_img);
            #endif
            undistort(or_img, or_img, camera_matrix, dist_coeffs);

            cout << "undistort" << endl;

            // Display fixed image
            #ifdef DEBUG
                my_imshow("Fixed", or_img);
            #endif
        #endif

        //Convert from RGB to HSV= Hue-Saturation-Value
        Mat hsv_img;
        cvtColor(or_img, hsv_img, COLOR_BGR2HSV);
        cout << "Conversion done" << endl;
        // Display HSV image
        #ifdef DEBUG
            my_imshow("HSVimage", hsv_img);
            mywaitkey();
        #endif

        // Find black regions (filter on saturation and value)
        // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
        Mat black_mask;
        COUT(sett->blackMask)
        inRange(hsv_img, sett->blackMask.Low(), sett->blackMask.High(), black_mask);
        #ifdef DEBUG
            my_imshow("BLACK_filter", black_mask);
            mywaitkey();
        #endif

        // Find contours
        // https://stackoverflow.com/questions/44127342/detect-card-minarea-quadrilateral-from-contour-opencv
        vector< vector<Point> > contours, contours_approx, contours_approx_big, contours_approx_big2;
        vector< Point > approx_polygon, rect, rect2;

        // Process black mask
        #if CV_MAJOR_VERSION<4
        findContours(black_mask, contours, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE); // find external contours of each blob
        #else 
        findContours(black_mask, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // find external contours of each blob
        #endif

        Mat app_img=or_img.clone();
        drawContours(app_img, contours, -1, Scalar(0,170,220), 10, LINE_AA);
        destroyAllWindows();

        double area, area_max = -1000.0, area_max2 = -1000.0;
        //for series of points
        for (unsigned i=0; i<contours.size(); i++){
          approxPolyDP(contours[i], approx_polygon, 10, true); // fit a closed polygon (with less vertices) to the given contour, with an approximation accuracy (i.e. maximum distance between the original and the approximated curve) of 10
          contours_approx = {approx_polygon};

          area = contourArea(approx_polygon); //compute area
          if (area>AREA_MIN){
            app_img=or_img.clone();
            if(area > area_max){ //find the second area max (the big rectangle)
                area_max2 = area_max;
                contours_approx_big2 = contours_approx_big;
                rect2 = rect;
                area_max = area;
                contours_approx_big = contours_approx;
                rect = approx_polygon;
            } else if(area > area_max2){
                area_max2 = area;
                contours_approx_big2 = contours_approx;
                rect2 = approx_polygon;
            }
          }
        }

        // I take the largest rectangle if the second one is very small, otherwise the second one
        if(area_max*AREA_RATIO<area_max2){
            rect = rect2;
            contours_approx_big = contours_approx_big2;
        }
        //Compute the right vertexes from a vector of points.
        find_rect(rect, or_img.size().width, or_img.size().height);
        
        // show the quadrilateral found
        Mat quadrilateral_img = or_img.clone();
        drawContours(quadrilateral_img, contours_approx_big, -1, Scalar(0,170,220), 5, LINE_AA);
        #ifdef WAIT
            my_imshow("Find rectangle", quadrilateral_img);
            mywaitkey();
        #endif
        
        //sort of 4 points in clockwise order
        vector<Point> tmp;
        convexHull(rect, tmp, true);
        rect=tmp;
        
        //sort in clockwise order starting from the leftmost corner
        int min_x=100000, min_index=0;
        for(int i=0; i<4; i++){
            if(rect[i].x < min_x){
                min_x = rect[i].x;
                min_index = i;
            } else if(rect[i].x==min_x){
                if(rect[i].y < rect[min_index].y){
                    min_index = i;
                }
            }
        }
        switch(min_index){
            case 0: 
                swap(rect[3], rect[1]);          
            break;
            case 1: 
                swap(rect[0], rect[1]); 
                swap(rect[2], rect[3]); 
            break;
            case 2: 
                swap(rect[0], rect[2]);          
            break;
            case 3: 
                swap(rect[0], rect[3]); 
                swap(rect[1], rect[2]); 
            break;
        }

        //check that the first side is the shortest
        float avg0 = ( distance(rect[0],rect[1]) + distance(rect[2],rect[3]) )/2;
        float avg1 = ( distance(rect[1],rect[2]) + distance(rect[3],rect[0]) )/2;
        if(avg0>avg1){
            Point tmp = rect[0];
            for(int i=0; i<3; i++){
                rect[i] = rect[i+1];
            }
            rect[3] = tmp;
        }

        //create the high corners
        vector<Point> rectHigh;
        createPointsHigh(rect, rectHigh);

        //wrap the perspective
        static const int width = 1000;
        static const int height = (int)(width*1.5);

        int xm/*in*/ = 0, ym = 0;
        int xM/*ax*/ = width, yM = height;
        Mat corner_pixels =      (Mat_<float>(4,2) << rect[0].x, rect[0].y, rect[1].x, rect[1].y, rect[2].x, rect[2].y, rect[3].x, rect[3].y);
        Mat corner_high_pixels = (Mat_<float>(4,2) << rectHigh[0].x, rectHigh[0].y, rectHigh[1].x, rectHigh[1].y, rectHigh[2].x, rectHigh[2].y, rectHigh[3].x, rectHigh[3].y);
        Mat transf_pixels =      (Mat_<float>(4,2) << xm, ym, xM, ym, xM, yM, xm, yM);

        //create the transformation matrices
        Mat transformationFromHigh = getPerspectiveTransform(corner_high_pixels, transf_pixels);
        getConversionParameters(transformationFromHigh, false);

        Mat transf = getPerspectiveTransform(corner_pixels, transf_pixels);
        Mat unwarped_frame;
        warpPerspective(or_img, unwarped_frame, transf, Size(width, height));

        // select a region of interest
        Mat imgCrop;
        imgCrop = unwarped_frame(Rect(0, 0, width, height));

        if(_imgRead){
            // Store the cropped image to disk.
            string file = sett->mapsNames.get(f);
            string save_name = file.substr(0, file.find_last_of('.'))+"_UN"+file.substr(file.find_last_of('.'), -1);
            string save_location = (sett->mapsFolder.back()=='/' ? sett->mapsFolder : sett->mapsFolder+"/")+save_name;
            if (!sett->addUnMap(save_name)){
                cerr << "File already indexed." << endl;
            }
            imwrite(save_location, imgCrop);
            cout << "Unwrapped image saved to: " << save_location << endl;
        } else{
            cout << "Returned image from unwrapping\n";
            *img = imgCrop; //return
        }
        
    }
    sett->writeToFile();
return(0);
}

void createPointsHigh(const vector<Point> & rectLow, vector<Point> & rectHigh){
    rectHigh.resize(0);

    rectHigh.push_back(Point(305, 1025));
    rectHigh.push_back(Point(300, 80));
    rectHigh.push_back(Point(1620, 75));
    rectHigh.push_back(Point(1633, 992));

    double dist1 = 0, dist2 = 0;
    for(int i=0; i<4; i++){
        dist1 += distance(rectLow[i], rectHigh[i]);
        dist2 += distance(rectLow[i], rectHigh[(i+2)%4]);
    }
    if(dist2<dist1){
        swap( rectHigh[0], rectHigh[2] );
        swap( rectHigh[1], rectHigh[3] );
    } // else nothing
}

void find_rect(vector<Point>& _rect, const int& width, const int& height){
    Tuple<Point2<int> > rect;
    
    const Point2<int> tl=Point2<int>(0, 0); //Top left corner
    const Point2<int> tr=Point2<int>(width, 0); //Top right corner 
    const Point2<int> bl=Point2<int>(0, height); //Bottom left corner
    const Point2<int> br=Point2<int>(width, height); //Bottom right corner

    //I start with all rect points centered
    Point2<int> r_tl=Point2<int>((int)(width/2), (int)(height/2)); //Top left corner
    Point2<int> r_tr=Point2<int>((int)(width/2), (int)(height/2)); //Top right corner 
    Point2<int> r_bl=Point2<int>((int)(width/2), (int)(height/2)); //Bottom left corner
    Point2<int> r_br=Point2<int>((int)(width/2), (int)(height/2)); //Bottom right corner

    //Compute intial distances
    double d_tl=r_tl.distance(tl);
    double d_tr=r_tr.distance(tr);
    double d_bl=r_bl.distance(bl);
    double d_br=r_br.distance(br);

    for (auto el : _rect){
        double _d_tl=Point2<int>(el).distance(tl);
        double _d_tr=Point2<int>(el).distance(tr);
        double _d_bl=Point2<int>(el).distance(bl);
        double _d_br=Point2<int>(el).distance(br);

        if (_d_tl < d_tl){
            r_tl=Point2<int>(el);
            d_tl=_d_tl;
        }
        if (_d_tr < d_tr){
            r_tr=Point2<int>(el);
            d_tr=_d_tr;
        }
        if (_d_bl < d_bl){
            r_bl=Point2<int>(el);
            d_bl=_d_bl;
        }
        if (_d_br < d_br){
            r_br=Point2<int>(el);
            d_br=_d_br;
        }
    }

    _rect.clear();
    _rect={Point(r_tl.x(), r_tl.y()), Point(r_tr.x(), r_tr.y()), Point(r_bl.x(), r_bl.y()), Point(r_br.x(), r_br.y())};
}

/*! \brief Load coefficients from a file.
    \details Load two matrix 'camera_matrix' and 'distortion_coefficients' from the xml file passed.
    \param[in] filename The string that identify the location of the xml file.
    \param[out] camera_matrix Where the 'camera_matrix' matrix is saved.
    \param[out] dist_coeffs Where the 'distortion_coefficients' matrix is saved.
*/
void loadCoefficients(  const string filename, 
                        Mat & camera_matrix, 
                        Mat & dist_coeffs)
{
    FileStorage fs(filename, FileStorage::READ );
    if (!fs.isOpened()){
        throw runtime_error("Could not open file " + filename);
    }

    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();
}

/*! \brief Compute the euclidean distance.

    \param[in, out] c1 The first point.
    \param[in, out] c2 The second point.

    \returns The euclidean distance.
*/
static double distance(Point c1, Point c2){
    double res = sqrt( pow( c2.x-c1.x ,2) + pow( c2.y-c1.y ,2) );
    return(res);
}
