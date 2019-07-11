#include"unwrapping.hh"

//debug blocks most things, wait only something
// #define DEBUG
// #define WAIT

#define area_ratio 0.7

const string xml_settings = "data/settings.xml";

static float distance(Point c1, Point c2);

/*! \brief Take some images according to a xml and unwrap the black rectangle inside the image after appling undistortion trasformation.
    \details Load from the xml file 'data/settings.xml' the name of some images, load the images from the file,\n
    apply the calibration (undistortion trasformation) thanks to the matrices load with the 'loadCoefficients' function.\n
    Then, with the use of a filter for the black the region of interest (a rectangle) is identified and all the perspective is rotated for reach a top view of the rectangle.\n
    Finally, the images are saved on some files.

    \returns A 0 is return if the function reach the end.
*/
int unwrapping(){
    FileStorage fs_xml(xml_settings, FileStorage::READ);

    const string calib_file = (string) fs_xml["calibrationFile"];
    for(unsigned f=0; f<fs_xml["mapsNames"].size(); f++){
        string filename = (string) fs_xml["mapsNames"][f];
        cout << "Elaborating of the file: " << filename << endl << endl;
        // Load image from file
        Mat or_img = imread(filename.c_str());
        
        // Display original image
        #ifdef WAIT
            my_imshow("Original", or_img, true);
        #endif

        // fix calibration with matrix
        Mat camera_matrix, dist_coeffs;
        loadCoefficients(calib_file, camera_matrix, dist_coeffs);

        
        Mat fix_img;
        undistort(or_img, fix_img, camera_matrix, dist_coeffs);

        // Display fixed image
        #ifdef DEBUG
            my_imshow("Fixed", fix_img);
        #endif

        //Convert from RGB to HSV= Hue-Saturation-Value
        Mat hsv_img;
        cvtColor(fix_img, hsv_img, COLOR_BGR2HSV);

        // Display HSV image
        #ifdef DEBUG
            my_imshow("HSVimage", hsv_img);
        #endif

        // Find black regions (filter on saturation and value)
        // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
        FileNode Bm = fs_xml["blackMask"];
        Mat black_mask;
        inRange(hsv_img, Scalar(Bm[0], Bm[1], Bm[2]), Scalar(Bm[3], Bm[4], Bm[5]), black_mask);
        #ifdef DEBUG
            my_imshow("BLACK_filter", black_mask);
        #endif
        
        // Find contours
        // https://stackoverflow.com/questions/44127342/detect-card-minarea-quadrilateral-from-contour-opencv
        vector< vector<Point> > contours, contours_approx, contours_approx_big, contours_approx_big2;
        vector< Point > approx_polygon, rect, rect2;

        // Process black mask
        findContours(black_mask, contours, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE); // find external contours of each blob

        double area, area_max = -1000.0, area_max2 = -1000.0;
        //for series of points
        for (unsigned i=0; i<contours.size(); i++){
            approxPolyDP(contours[i], approx_polygon, 10, true); // fit a closed polygon (with less vertices) to the given contour, with an approximation accuracy (i.e. maximum distance between the original and the approximated curve) of 10
            contours_approx = {approx_polygon};

            //if we find a quadrilateral -> compute the max area
            if(approx_polygon.size()==4){
                area = contourArea(approx_polygon); //compute area
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
        if(area_max*area_ratio<area_max2){
            rect = rect2;
            contours_approx_big = contours_approx_big2;
        }
        
        // show the quadrilateral found
        Mat quadrilateral_img = fix_img.clone();
        drawContours(quadrilateral_img, contours_approx_big, -1, Scalar(0,170,220), 5, LINE_AA);
        #ifdef WAIT
            my_imshow("Find rectangle", quadrilateral_img);
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

        // wrap the perspective
        static const int width = 1000;
        static const int height = (int)(width*1.5);

        int xm/*in*/ = 0, ym = 0;
        int xM/*ax*/ = width, yM = height;
        Mat corner_pixels = (Mat_<float>(4,2) << rect[0].x, rect[0].y, rect[1].x, rect[1].y, rect[2].x, rect[2].y, rect[3].x, rect[3].y);
        Mat transf_pixels = (Mat_<float>(4,2) << xm, ym, xM, ym, xM, yM, xm, yM);

        Mat transf = getPerspectiveTransform(corner_pixels, transf_pixels);

        Mat unwarped_frame;
        warpPerspective(fix_img, unwarped_frame, transf, Size(width, height));

        // select a region of interest
        Mat imgCrop;
        // imgCrop = unwarped_frame(Rect(xm, ym, xM-xm, yM-ym));
        imgCrop = unwarped_frame(Rect(0, 0, width, height));
        #ifdef WAIT
            my_imshow("cropped image", imgCrop);
        #endif

        // Store the cropped image to disk.
        string save_location = (string) fs_xml["mapsUnNames"][f];
        imwrite(save_location, imgCrop);

        #ifdef WAIT
            // wait a char 'q' to proceed
            while((char)waitKey(1)!='q'){}
        #endif
        
    }
return(0);
}

/*! \brief Load coefficients from a file.
    \details Load two matrix 'camera_matrix' and 'distortion_coefficients' from the xml file passed.
    \param[in] filename The string that identify the location of the xml file.
    \param[out] camera_matrix Where the 'camera_matrix' matrix is saved.
    \param[out] dist_coeffs Where the 'distortion_coefficients' matrix is saved.
*/
void loadCoefficients(  const string filename, 
                        Mat & camera_matrix, 
                        Mat & dist_coeffs){
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
static float distance(Point c1, Point c2){
    float res = sqrt( pow( c2.x-c1.x ,2) + pow( c2.y-c1.y ,2) );
    return(res);
}