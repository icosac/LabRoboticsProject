//compilation command: g++ `pkg-config --cflags opencv` -Wall -O3 -o create_xml.out create_xml.cc `pkg-config --libs opencv`

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(){
    string filename[2] = {"data/settings.yaml", "data/settings.xml"};
    for(int i=0; i<2; i++){
        //__________________________________________write__________________________________________
        FileStorage fs(filename[i], FileStorage::WRITE);
        cout << "\tWrite on " << filename[i] << endl;

        // map
        fs << "mapsNames" << "[";                              // text - string sequence
        fs << "./data/map/03.jpg" << "./data/map/02.jpg" << "./data/map/01.jpg";
        fs << "]";                                            // close sequence
        
        fs << "calibrationFile" << "./data/intrinsic_calibration.xml";

        //--------------> mask
        //black
        fs << "blackMask" << "[" << 0 << 0 << 0 << 180 << 255 << 100 << "]";
        //fs << "blackMaskLow" << Scalar(0, 0, 0);
        //fs << "blackMaskHigh" << Scalar(180, 255, 100);

        fs.release();                                       // explicit close
        
        //__________________________________________read__________________________________________
        /*FileStorage FS;
        FS.open(filename[i], FileStorage::READ);
        cout << "\tRead on " << filename[i] << endl;

        FileNode n = FS["mapsNames"];                    // Read string sequence - Get node

        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for(; it != it_end; it++){  //I do the comparison between two pointers (two addresses)
            cout << (string)*it << endl;
        }

        cout << (string) FS["calibrationFile"] << endl;

        FileNode Bm = FS["blackMask"];
        for(int i=0; i<6; i++){
            cout << (int) Bm[i] << " ";
        }
        cout << endl << endl;*/
    }
    return 0;
}