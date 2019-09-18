#include <robotProject.hh>
#include <iostream>
#include <unistd.h>
#include <iostream>
#include "path.h"

using namespace std;

extern Settings *sett;
extern CameraCapture* camera;

int main(){ 
    sett->cleanAndRead(FILE);
    
    double frame_time=0;
    RobotProject rp= RobotProject(nullptr, frame_time);
    cout << *sett << endl;

    cout << sett->unMaps(0).get(0) << endl;
    Mat img=imread(sett->unMaps(0).get(0));
    if(!rp.preprocessMap(img)){
        cout << "Error1\n";
    }

    Path path;
    if(!rp.planPath(img, path)){
        cout << "Error2\n";
    }
    cout << "path size: " << path.size() << endl;

    cout << "END\n";
    return(0);   
}