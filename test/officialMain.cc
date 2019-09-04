#include <robotProject.hh>
#include <iostream>
#include <unistd.h>
#include <iostream>

using namespace std;

Settings *sett = new Settings();

int main(){ 
    sett->cleanAndRead();
    cout << "Official Main:\n";
    RobotProject rp=RobotProject();

    Mat img = acquireImage(true);
    // Mat img = imread(sett->maps(0).get(0).c_str());
    COUT(*sett)
    if(!rp.preprocessMap(img)){
        cout << "Error1\n";
    }

    ClipperLib::Path path;
    if(!rp.planPath(img, path)){
        cout << "Error2\n";
    }
    cout << "path size: " << path.size() << endl;

    Mat imgNew = acquireImage(true);
    // Mat imgNew = imread(sett->maps(0).get(0).c_str());
    vector<double> state; //x, y, th
    for(int i=0; i<3; i++){
        cout << i+1 << "° ";
        
        if(!rp.localize(imgNew,  state)){
            cout << "Error3\n";
        }
        sleep(0.05);
    }

return(0);   
}