#include"unwrapping.hh"

const string filename = "./data/map/03.jpg";
const string calib_file = "./data/intrinsic_calibration.xml";

int unwrapping(){
    // Load image from file
    Mat or_img = imread(filename.c_str());
    if(or_img.empty()) {
        throw runtime_error("Failed to open the file " + filename);
    }
    
    // Display original image
    my_imshow("Original", or_img);

    // fix calibration with matrix
    Mat camera_matrix, dist_coeffs;
    loadCoefficients(calib_file, camera_matrix, dist_coeffs);

    Mat fix_img;
    undistort(or_img, fix_img, camera_matrix, dist_coeffs);
    
    // Display fixed image
    my_imshow("Fixed", fix_img);

    //Convert from RGB to HSV= Hue-Saturation-Value
    Mat hsv_img;
    cvtColor(fix_img, hsv_img, COLOR_BGR2HSV);

    // Display HSV image
    my_imshow("HSVimage", hsv_img);

    // Find black regions (filter on saturation and value)
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    Mat black_mask;
    inRange(hsv_img, Scalar(0, 0, 0), Scalar(180, 255, 100), black_mask);  
    my_imshow("BLACK_filter", black_mask);
    
    // Find contours
    // https://stackoverflow.com/questions/44127342/detect-card-minarea-quadrilateral-from-contour-opencv
    vector< vector<Point> > contours, contours_approx, contours_approx_big;
    vector< Point > approx_polygon, rect;

    // Process black mask
    findContours(black_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // find external contours of each blob

    //cout << "N. contours: " << contours.size() << endl;
    double area, area_max = -1000;
    //for series of points
    for (int i=0; i<contours.size(); ++i){
        approxPolyDP(contours[i], approx_polygon, 10, true); // fit a closed polygon (with less vertices) to the given contour, with an approximation accuracy (i.e. maximum distance between the original and the approximated curve) of 3
        contours_approx = {approx_polygon};

        //if we find a quadrilateral
        if(approx_polygon.size()==4){
            //cout << (i+1) << ") Contour size: " << contours[i].size() << endl;
            //cout << "   Approximated contour size: " << approx_polygon.size() << endl << approx_polygon << endl;

            area = contourArea(approx_polygon); //compute area
            if(area > area_max){ //find area max (the big rectangle)
                area_max = area;
                contours_approx_big = contours_approx;
                rect = approx_polygon;
            }
        }
    }
    cout << "area max: " << area << endl;
    
    // show the quadrilateral found
    Mat quadrilateral_img = fix_img.clone();
    drawContours(quadrilateral_img, contours_approx_big, -1, Scalar(0,170,220), 5, LINE_AA);
    my_imshow("Find rectangle", quadrilateral_img);
    
    //sort of 4 points in counterclockwise order
    PrintPoints("input: ", rect);
    Sort4PointsCounterClockwise(rect);
    PrintPoints("output:undistort ", rect);
    printf("\n");
    
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
    int support;
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
    PrintPoints("output: ", rect);

    //check that the first side is the shortest
    distance(rect[0],rect[1]);
    distance(rect[1],rect[2]);
    distance(rect[2],rect[3]);
    distance(rect[3],rect[0]);
    cout <<"\n----------------------------------\n";
    float avg0 = ( distance(rect[0],rect[1]) + distance(rect[2],rect[3]) )/2;
    float avg1 = ( distance(rect[1],rect[2]) + distance(rect[3],rect[0]) )/2;
    if(avg0>avg1){
    //if(distance(rect[0],rect[1])>distance(rect[1],rect[2])){  //far  sides
    //if(distance(rect[2],rect[3])>distance(rect[3],rect[0])){//near sides
        cout << "\navg0 > avg1\n";
        Point tmp = rect[0];
        for(int i=0; i<3; i++){
            rect[i] = rect[i+1];
        }
        rect[3] = tmp;
    }else{
        cout << "\navg0 < avg1\n";
    }
    PrintPoints("output: ", rect);

    // wrap the perspective
    Mat corner_pixels = (Mat_<float>(4,2) << rect[0].x, rect[0].y, rect[1].x, rect[1].y, rect[2].x, rect[2].y, rect[3].x, rect[3].y);
    Mat transf_pixels = (Mat_<float>(4,2) << 50, 50, 550, 50, 550, 800, 50, 800);
    cout << "corner_pixels:\n" << corner_pixels << endl; 
    cout << "transf_pixels:\n" << transf_pixels << endl;

    Mat transf = getPerspectiveTransform(corner_pixels, transf_pixels);
    cout << "transf:\n" << transf << endl;

    Mat unwarped_frame;
    warpPerspective(fix_img, unwarped_frame, transf, fix_img.size());

    // select a region of interest
    Mat imgCrop = unwarped_frame(Rect(0, 0, 600, 850));
    namedWindow("cropped image", CV_WINDOW_NORMAL);
    my_imshow("cropped image", imgCrop);   

    // wait a char 'q' to proceed
    while((char)waitKey(1)!='q'){}
return(0);
}


void loadCoefficients(const string& filename, Mat& camera_matrix, Mat& dist_coeffs){
  FileStorage fs(filename, FileStorage::READ );
  if (!fs.isOpened()){
    throw runtime_error("Could not open file " + filename);
  }
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> dist_coeffs;
  fs.release();
}

void my_imshow(const char*  win_name, Mat img){
    const int SIZE     = 330;
    const int W_0      = 0;
    const int H_0      = 0;
    const int W_OFFSET = 20;
    const int H_OFFSET = 90;
    const int LIMIT    = W_0 + 4*SIZE + 3*W_OFFSET;

    static int W_now = W_0;
    static int H_now = H_0;

    //const string s = win_name;
    namedWindow(win_name, CV_WINDOW_NORMAL);
    cvvResizeWindow(win_name, SIZE, SIZE);
    imshow(win_name, img);
    moveWindow(win_name, W_now, H_now);
    //cout << W_now << " " << H_now << endl;
    W_now += SIZE + W_OFFSET;
    if(W_now >= LIMIT){
        W_now = W_0;
        H_now += SIZE + H_OFFSET;
    }
}

float distance(Point c1, Point c2){
    float res = sqrt( pow( c2.x-c1.x ,2) + pow( c2.y-c1.y ,2) );
    cout << res << "\t";
    return(res);
}

void swap(int & a, int & b){
    int c = a;
    a = b;
    b = c;
}

//___________________________   SORT POINTS  ___________________________________________________________________________
//https://stackoverflow.com/questions/242404/sort-four-points-in-clockwise-order/242509#242509
// Returns the z-component of the cross product of a and b
double CrossProductZ(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}

// Orientation is positive if abc is counterclockwise, negative if clockwise.
// (It is actually twice the area of triangle abc, calculated using the
// Shoelace formula: http://en.wikipedia.org/wiki/Shoelace_formula .)
double Orientation(const Point &a, const Point &b, const Point &c) {
    return CrossProductZ(a, b) + CrossProductZ(b, c) + CrossProductZ(c, a);
}

void Sort4PointsCounterClockwise(vector<Point_<int> > & points){
    Point& a = points[0];
    Point& b = points[1];
    Point& c = points[2];
    Point& d = points[3];

    if (Orientation(a, b, c) < 0.0) {
        // Triangle abc is already clockwise.  Where does d fit?
        if (Orientation(a, c, d) < 0.0) {
            return;           // Cool!
        } else if (Orientation(a, b, d) < 0.0) {
            swap(d, c);
        } else {
            swap(a, d);
        }
    } else if (Orientation(a, c, d) < 0.0) {
        // Triangle abc is counterclockwise, i.e. acb is clockwise.
        // Also, acd is clockwise.
        if (Orientation(a, b, d) < 0.0) {
            swap(b, c);
        } else {
            swap(a, b);
        }
    } else {
        // Triangle abc is counterclockwise, and acd is counterclockwise.
        // Therefore, abcd is counterclockwise.
        swap(a, c);
    }
}

void PrintPoints(const char *caption, const vector<Point_<int> > & points){
    printf("%s: (%d,%d),(%d,%d),(%d,%d),(%d,%d)\n", caption,
        points[0].x, points[0].y, points[1].x, points[1].y,
        points[2].x, points[2].y, points[3].x, points[3].y);
}