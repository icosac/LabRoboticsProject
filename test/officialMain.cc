#include <robotProject.hh>
#include <iostream>
#include <unistd.h>
#include <iostream>
#include "path.h"

using namespace std;

extern Settings *sett;
extern CameraCapture* camera;

int main(){ 
    sett->cleanAndRead();
    
    cout << "Official Main:\n";
    RobotProject rp=RobotProject(true);

    Mat img = acquireImage(true);
    // Mat img = imread(sett->maps(0).get(0).c_str());
    COUT(*sett)
    if(!rp.preprocessMap(img)){
        cout << "Error1\n";
    }

    Path path;
    if(!rp.planPath(img, path)){
        cout << "Error2\n";
    }
    cout << "path size: " << path.size() << endl;

    Mat imgNew = imread(sett->maps(0).get(0).c_str());
    vector<double> state; //x, y, th
    double frame_time;
    cout << "capturing" << endl;
    for(int i=0; i<5; i++){
        Mat imgNew;

        camera->grab(imgNew, frame_time);
        imwrite(("/home/robotics/Desktop/imgNew"+to_string(i)+".jpg").c_str(), imgNew);
        cout << i+1 << "° ";
        
        if(!rp.localize(imgNew,  state)){
            cout << "Error3\n";
        }
        sleep(2);
    }

    cout << "END\n";
return(0);   
}