#include <robotProject.hh>
#include <iostream>
#include <unistd.h>
using namespace std;

int main(){ 
    cout << "Official Main:\n";
    RobotProject rp=RobotProject();

    Mat img;
    if(!rp.preprocessMap(img)){
        cout << "Error1\n";
    }

    ClipperLib::Path path;
    if(!rp.planPath(img, path)){
        cout << "Error2\n";
    }

    Mat imgNew;
    vector<double> state;
    for(int i=0; i<5; i++){
        cout << i+1 << "° ";
        
        if(!rp.localize(imgNew,  state)){
            cout << "Error3\n";
        }
        sleep(0.05);
    }

return(0);   
}