#include <map.hh>
#include <iostream>
#include <maths.hh>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int main(){
    cout << "MAIN MAP\n";
    int dimX=300, dimY=450;
    Mapp* map = new Mapp(dimX, dimY, 5, 5);

    vector<Point2<int> > vp;
    vp.push_back(Point2<int>(13, 7));
    vp.push_back(Point2<int>(32, 7));
    //vp.push_back(Point2<int>(30, 20));
    vp.push_back(Point2<int>(14,18));

    map->addObject(vp, OBST);
    map->printMap();
    
    /*Point2<int> p = Point2<int>(26, 16);
    cout << "Checked point: " << p << " -> " << map->getPointType(p) << endl;

    cout << "Checked segment 1: " << map->checkSegment(p, Point2<int>(7, 7)) << endl;
    cout << "Checked segment 2: " << map->checkSegment(p, Point2<int>(18, 33)) << endl;*/


    // real time localization and printing of the map
    Mat imageMap = map->createMapRepresentation();
    namedWindow("Map", WINDOW_AUTOSIZE);
    int x = rand()%dimX;
    int y = rand()%dimY;

    // the wait 200 simulate the call every x milliseconds
	while((char)waitKey(200)!='x'){ // wait a char 'x' to proceed
        //Point p = localize();
        x = (x+(rand()%10))%dimX;
        y = (y+(rand()%10))%dimY;
        // cout << x << " " << y << endl;
        circle(imageMap, Point(x, y), 5, Scalar(0, 255, 255), -1);
        imshow("Map", imageMap);
	}
return(0);
}   