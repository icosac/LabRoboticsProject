#include "detection.hh"

const string xml_settings = "data/settings.xml";
FileStorage fs_xml;
vector<Mat> templates;

// #define DEBUG
// #define WAIT

/*! \brief Loads some images and detects shapes according to different colors.

    \returns Return 0 if the function reach the end.
*/
int detection(){

    fs_xml.open(xml_settings, FileStorage::READ);

    load_number_template();

    for(unsigned f=0; f<1 && f<fs_xml["mapsUnNames"].size(); f++){
        string filename = (string) fs_xml["mapsUnNames"][f];
        cout << "Elaborating file: " << filename << endl;

        // Load unwrapped image from file
        Mat un_img = imread(filename.c_str());
        if(un_img.empty()) {
            throw runtime_error("Failed to open the image " + filename);
        }

        #ifdef WAIT
            my_imshow("unwrapped image", un_img, true);
        #endif
        
        //Convert from RGB to HSV
        Mat hsv_img;
        cvtColor(un_img, hsv_img, COLOR_BGR2HSV);

        //detection (red-green-blue)
        for(int i=0; i<3; i++){
            shape_detection(hsv_img, i);
            #ifdef WAIT
                if(i!=2){
                    waitKey();
                    destroyAllWindows();
                    my_imshow("unwrapped image", un_img, true);
                }
            #endif
        }
        
        #ifdef WAIT
            // wait a char 'q' to proceed
            while((char)waitKey(1)!='q'){}
        #endif
    }
    return(0);
}

/*! \brief Load some templates and save them in the global variable 'templates'.
*/
void load_number_template(){ //load the template for number recognition
    string folder = fs_xml["templatesFolder"];
    const int n_template = fs_xml["templates"].size();
    string tmp_str;
    Mat tmp;
    for(int i=0; i<n_template; i++){
        fs_xml["templates"][i] >> tmp_str;
        tmp = imread(folder + tmp_str);
        cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
        bitwise_not(tmp, tmp);
        templates.push_back(tmp);
    }
}

/*! \brief Detect shapes inside the image according to the variable 'color'.

    \param[in] img Image on which the research will done.
    \param[in] color Can has 3 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    These color identify the possible spectrum that the function search on the image.
*/
void shape_detection(const Mat & img, const int color){
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    FileNode mask;
    switch(color){
        case 0: 
            cout << "\tObstacles detection\n";  
            mask = fs_xml["redMask"];
        break;
        case 1:  
            cout << "\tVictim detection\n";   
            mask = fs_xml["greenMask"];  
        break;
        case 2:  
            cout << "\tGate detection\n";    
            mask = fs_xml["blueMask"];   
        break;
    }
    
    Mat color_mask;
    if(color==0){
        Mat red_mask_low, red_mask_high;
        inRange(img, Scalar(0, mask[1], mask[2]), Scalar(mask[0], mask[4], mask[5]), red_mask_low);  
        inRange(img, Scalar(mask[3], mask[1], mask[2]), Scalar(179, mask[4], mask[5]), red_mask_high);
        addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0, color_mask); // combine together the two binary masks
    } else{
        inRange(img, Scalar(mask[0], mask[1], mask[2]), Scalar(mask[3], mask[4], mask[5]), color_mask);
    }
    #ifdef DEBUG
        my_imshow("Color_filter", color_mask);
    #endif

    erode_dilation(color_mask, color);
    #ifdef WAIT
        my_imshow("Color filtered", color_mask);
    #endif

    find_contours(color_mask, img, color);
}

/*! \brief It apply some filtering function for isolate the subject and remove the noise.
    \details An example of the sub functions called are: GaussianBlur, Erosion, Dilation and Threshold.

    \param[in, out] img Is the image on which the function apply the filtering.
    \param[in] color Can has 4 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    3 -> Black\n
    According to the color the filtering functions apply can change in the type and in the order.
*/
void erode_dilation(Mat & img, const int color){
    const int erode_side = (int) fs_xml["kernelSide"]; //odd number
    const int center = erode_side/2+1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(erode_side, erode_side), Point(center, center) );
    // 0red && 2blue    -> smooth - erode - dilation - smooth - treshold
    // 1green && 3black -> smooth - dilation - smooth - erode - treshold

    //smooth -> gaussian blur
    GaussianBlur(img, img, Size(erode_side, erode_side), 1, 1);
    //my_imshow("Smooth 1", img);

    if(color==0 || color==2 || color==3){
        // Apply the erode operation
        erode(img, img, kernel);
        //my_imshow("Erode", img);
    }

    // Apply the dilation operation
    dilate(img, img, kernel);
    //my_imshow("Dilation", img);

    //smooth -> gaussian blur
    GaussianBlur(img, img, Size(erode_side, erode_side), 1, 1);
    //my_imshow("Smooth 2", img);

    if(color==1){
        // Apply the erode operation
        erode(img, img, kernel);
        //my_imshow("Erode", img);
    }

    threshold(img, img, 100, 255, 0 ); // threshold and binarize the image, to suppress some noise
    //my_imshow("treshold", img);
}

/*! \brief Given an image, in black/white format, identify all the borders that delimit the shapes.
    
    \param[in] img Is an image in HSV format at the base of the elaboration process.
    \param[out] original Is the original source of 'img', it is used for showing the detected contours.
    \param[in] color Can has 3 value:\n
    0 -> Red\n
    1 -> Green\n
    2 -> Blue\n
    Is used for decid which procedure apply to the image.
*/
void find_contours( const Mat & img, 
                    Mat original, 
                    const int color){
    const double MIN_AREA_SIZE = 100;
    vector<vector<Point>> contours, contours_approx;
    vector<Point> approx_curve;
    vector<int> victimNum;
    
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // find external contours of each blob
    drawContours(original, contours, -1, Scalar(40,190,40), 1, LINE_AA);
    
    if(color==1){cout << "\tNumber detection\n";}
    for (unsigned i=0; i<contours.size(); ++i){
        if (contourArea(contours[i]) > MIN_AREA_SIZE){ // filter too small contours to remove false positives
            approxPolyDP(contours[i], approx_curve, 3, true); // fit a closed polygon (with less vertices) to the given contour, with an approximation accuracy (i.e. maximum distance between the original and the approximated curve) of 3
            
            if(color==1){ //green
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
    drawContours(original, contours_approx, -1, Scalar(0,170,220), 5, LINE_AA);
    #ifdef WAIT
        my_imshow("Detected shape", original);
    #endif
    save_convex_hull(contours_approx, color, victimNum);
}

/*! \brief Given some vector save it in a xml file.

    \param[in] contours Is a vector that is saved in a xml file.
    \param[in] color Is the parameter according to which the function decide if saved ('color==1') or not ('otherwise') the vector 'victims'.
    \param[in] victims Is a vector that is saved in a xml file.
*/
void save_convex_hull(  const vector<vector<Point>> & contours, 
                        const int color, 
                        const vector<int> & victims){
    vector<vector<Point>> hull;
    vector<Point> hull_i;
    for(unsigned i=0; i<contours.size(); i++){
        convexHull(contours[i], hull_i, true);//return point in clockwise order
        hull.push_back(hull_i);
    }
    string save_file = fs_xml["convexHullFile"];
    static FileStorage fs(save_file, FileStorage::WRITE);
    string str;
    switch(color){
        case 0: str="obstacles"; break;
        case 1: str="victims"; break;
        case 2: str="gate"; break;
    }
    fs << str << hull;
    if(color==1){
        fs << "victimsNum" << victims;
    }
    //fs.release(); //if I do this operation I save only the first call of this function...
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
    FileNode mask = fs_xml["blackMask"];
    inRange(processROI, Scalar(mask[0], mask[1], mask[2]), Scalar(mask[3], mask[4], mask[5]), processROI);
    
    erode_dilation(processROI, 3);
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
        matchTemplate(processROI, templates[i], result, TM_CCOEFF); //TM_SQDIFF
        double score;
        minMaxLoc(result, nullptr, &score);
        //my_imshow("templates[i]", templates[i]);
        //cout << i << " score of " << score << endl;
        //waitKey();
        if (score > maxScore) {
            maxScore = score;
            maxIdx = i;
        }
    }
    #ifdef DEBUG
        cout << "Best fitting template: -> " << maxIdx << " <- with score of: " << maxScore << endl << endl;
        waitKey();
    #endif
    return(maxIdx%10);  //if we have 20-30-... templates it return the true number
}

/*! \brief Given an image identify the region of interest(ROI) and crop it out. 

    \param[in,out] ROI Is the image that the function will going to elaborate.
*/
void crop_number_section(Mat & ROI){
    // Tutorial for the min rectangle arround a shape. https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rotated_ellipses/bounding_rotated_ellipses.html
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
        convexHull(tmpContour, contour, true);//return point in clockwise order
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
    //resize(ROI, ROI, ROI.size());

    #ifdef DEBUG
        my_imshow("rotated Num", ROI);
    #endif
}