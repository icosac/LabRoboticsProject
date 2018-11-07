#include "detection.hh"

// #ifdef DEBUG
//     #define DEBUG (arg) printf("%s", arg)
// #else 
//     #define DEBUG ()
// #endif

const string xml_settings = "data/settings.xml";
FileStorage fs_xml;
vector<Mat> templates;

#define DEBUG

int detection(){
    INFO("ciaone")
    fs_xml.open(xml_settings, FileStorage::READ);

    load_number_template();

    for(unsigned f=0; f<1/*fs_xml["mapsUnNames"].size()*/; f++){
        string filename = (string) fs_xml["mapsUnNames"][f];
        cout << "Elaborating file: " << filename << endl;

        // Load unwrapped image from file
        Mat un_img = imread(filename.c_str());
        if(un_img.empty()) {
            throw runtime_error("Failed to open the image " + filename);
        }

        my_imshow("unwrapped image", un_img, true);
        
        //Convert from RGB to HSV
        Mat hsv_img;
        cvtColor(un_img, hsv_img, COLOR_BGR2HSV);

        //detection (red-green-blue)
        for(int i=0; i<3; i++){
            shape_detection(hsv_img, i);
            if(i!=2){
                waitKey();
                destroyAllWindows();
                my_imshow("unwrapped image", un_img, true);
            }
        }
        
        // wait a char 'q' to proceed
        while((char)waitKey(1)!='q'){}
    }
    return(0);
}

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

void shape_detection(Mat img, const int color){
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
    my_imshow("Color_filter", color_mask);

    erode_dilation(color_mask, color);
    my_imshow("Color filtered", color_mask);

    find_contours(color_mask, img, color);
}

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

void find_contours(const Mat & img, Mat original, const int color){
    const double MIN_AREA_SIZE = 100;
    vector<vector<Point>> contours, contours_approx;
    vector<Point> approx_curve;
    vector<int> victimNum;
    
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // find external contours of each blob
    drawContours(original, contours, -1, Scalar(40,190,40), 1, LINE_AA);
    //cout << "N. contours: " << contours.size() << endl;
    for (unsigned i=0; i<contours.size(); ++i){
        if (contourArea(contours[i]) > MIN_AREA_SIZE){ // filter too small contours to remove false positives
            //cout << (i+1) << ") Contour size: " << contours[i].size() << endl;
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
    //cout << endl;
    drawContours(original, contours_approx, -1, Scalar(0,170,220), 5, LINE_AA);
    my_imshow("Original", original);
    save_convex_hull(contours_approx, color, victimNum);
}

void save_convex_hull(vector<vector<Point>> & contours, const int color, vector<int> victims){
    vector<vector<Point>> hull;
    vector<Point> hull_i;
    for(unsigned i=0; i<contours.size(); i++){
        //cout << "contours: size " << contours[i].size() << "\n" << contours[i] << endl;
        convexHull(contours[i], hull_i, true);//return point in clockwise order
        hull.push_back(hull_i);
        //cout << "hull: size " << hull[i].size() << "\n" << hull[i] << endl << endl << endl;
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

    contours.resize(0);
}

int number_recognition(Rect blob, const Mat & base){ //filtering
    cout << "\tNumber detection\n"; 
    Mat processROI(base, blob); // extract the ROI containing the digit
    if(processROI.empty()){return(-1);}
    
    resize(processROI, processROI, Size(200, 200)); // resize the ROI
    my_imshow("Resize num", processROI);

    // black filter
    FileNode mask = fs_xml["blackMask"];
    inRange(processROI, Scalar(mask[0], mask[1], mask[2]), Scalar(mask[3], mask[4], mask[5]), processROI);
    my_imshow("ROI mask", processROI);
    
    erode_dilation(processROI, 3);
    my_imshow("ROI filtered", processROI);

    //vector<vector<Point>> contours
    //findContours(processROI, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //RotatedRect ellipse = fitEllipse(contours[0]);


    #ifndef TESSERACT
        // matching template
        // Find the template digit with the best matching
        double maxScore = -5e10;//5e   8;  // I don't know what this number represents...
        int maxIdx = -1;
        for (unsigned j=0; j<templates.size(); j++) {
            Mat result;
            matchTemplate(processROI, templates[j], result, TM_CCOEFF); //TM_SQDIFF
            my_imshow("templates[j]", templates[j], false);
            waitKey();
            double score;
            minMaxLoc(result, nullptr, &score); 
            cout << j << ": " << score << endl;
            if (score > maxScore) {
                maxScore = score;
                maxIdx = j;
            }
        }
        cout << "Best fitting template: " << maxIdx << " with score of: " << maxScore << endl << endl;
        waitKey();
        return(maxIdx);
    #else
        //tesseract OCR setting
        cout << "\n\nTESSERACT\n\n";
        // Create Tesseract object
        tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
        // Initialize tesseract to use English (eng) 
        ocr->Init(NULL, "eng");
        // Set Page segmentation mode to PSM_SINGLE_CHAR (10)
        ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
        // Only digits are valid output characters
        ocr->SetVariable("tessedit_char_whitelist", "056789");
        
        //tesseract OCR running
        // Set image data
        ocr->SetImage(processROI.data, processROI.cols, processROI.rows, 3, processROI.step);
        // Run Tesseract OCR on image and print recognized digit
        cout << "Recognized digit: " << string(ocr->GetUTF8Text()) << endl;

        ocr->End(); // destroy the ocr object (release resources)*/
    #endif
}











/*
void obstacle_detection(Mat img){ //R: 0 red
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    cout << "\tObstacles detection\n";
    FileNode Bm = fs_xml["redMask"];
    Mat red_mask_low, red_mask_high, red_mask;
    inRange(img, Scalar(0, Bm[1], Bm[2]), Scalar(Bm[0], Bm[4], Bm[5]), red_mask_low);  
    inRange(img, Scalar(Bm[3], Bm[1], Bm[2]), Scalar(179, Bm[4], Bm[5]), red_mask_high);
    addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0, red_mask); // combine together the two binary masks
    my_imshow("RED_filter", red_mask);

    erode_dilation(red_mask, 0);
    my_imshow("RED filtered", red_mask);

    find_contours(red_mask, img);
}

void victim_detection(Mat img){ //G: 1 green
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    cout << "\tVictim detection\n";
    FileNode Bm = fs_xml["greenMask"];
    Mat green_mask;
    inRange(img, Scalar(Bm[0], Bm[1], Bm[2]), Scalar(Bm[3], Bm[4], Bm[5]), green_mask);  
    my_imshow("GREEN_filter", green_mask);

    erode_dilation(green_mask, 1);
    my_imshow("GREEN filtered", green_mask);

    find_contours(green_mask, img);
}

void gate_detection(Mat img){ //B: 2 blue
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    cout << "\tGate detection\n";
    FileNode Bm = fs_xml["blueMask"];
    Mat blue_mask;
    inRange(img, Scalar(Bm[0], Bm[1], Bm[2]), Scalar(Bm[3], Bm[4], Bm[5]), blue_mask);  
    my_imshow("BLUE_filter", blue_mask);

    erode_dilation(blue_mask, 2);
    my_imshow("BLUE filtered", blue_mask);

    find_contours(blue_mask, img);
}

void contour_detection(Mat img){ //Bl: 3 black
    // HSV range opencv: Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    cout << "\tContour detection\n";
    FileNode Bm = fs_xml["blackMask"];
    Mat black_mask;
    inRange(img, Scalar(Bm[0], Bm[1], Bm[2]), Scalar(Bm[3], Bm[4], Bm[5]), black_mask);  
    my_imshow("BLACK_filter", black_mask);

    erode_dilation(black_mask, 3);
    my_imshow("BLACK filtered", black_mask);

    //find_contours(black_mask, img);
}*/

