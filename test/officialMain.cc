#include <robotProject.hh>
#include <iostream>
#include <unistd.h>
using namespace std;

Settings *sett = new Settings();

int main(){ 
    cout << "Official Main:\n";
    RobotProject rp=RobotProject();

    //Mat img = acquireImage(false);
    Mat img = imread(sett->maps(0).get(0).c_str());
    if(!rp.preprocessMap(img)){
        cout << "Error1\n";
    }

    ClipperLib::Path path;
    if(!rp.planPath(img, path)){
        cout << "Error2\n";
    }
    cout << "path size: " << path.size() << endl;

    Mat imgNew = imread(sett->maps(0).get(0).c_str());
    vector<double> state; //x, y, th
    for(int i=0; i<10; i++){
        cout << i+1 << "° ";
        
        if(!rp.localize(imgNew,  state)){
            cout << "Error3\n";
        }
        sleep(0.05);
    }

return(0);   
}