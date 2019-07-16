//compilation command: g++ `pkg-config --cflags opencv` -Wall -O3 -o create_xml.out create_xml.cc `pkg-config --libs opencv`

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(){
    string filename[2] = {"data/settings.xml", "data/settings.yaml"};
    for(int i=0; i<1; i++){ //write only on xml, yaml is redundant
        //__________________________________________write__________________________________________
        FileStorage fs(filename[i], FileStorage::WRITE);
        cout << "\tWrite on " << filename[i] << endl;

        /*/ map
        fs << "mapsNames" << "[";                              // text - string sequence
        fs << ".data/myMap/myMap_01.jpg" << ".data/myMap/myMap_02.JPG" << ".data/myMap/myMap_03.JPG" << ".data/myMap/myMap_04.JPG" << ".data/myMap/myMap_05.JPG" << ".data/myMap/myMap_06.JPG" << ".data/myMap/myMap_07.JPG" << ".data/myMap/myMap_08.JPG" << ".data/myMap/myMap_09.JPG" << ".data/myMap/myMap_10.JPG" << ".data/myMap/myMap_11.JPG";
        fs << "]";                                            // close sequence

        // map_unwrapped
        fs << "mapsUnNames" << "[";                              // text - string sequence
        fs << ".data/myMap/myMapUN_01.jpg" << ".data/myMap/myMapUN_02.JPG" << ".data/myMap/myMapUN_03.JPG" << ".data/myMap/myMapUN_04.JPG" << ".data/myMap/myMapUN_05.JPG" << ".data/myMap/myMapUN_06.JPG" << ".data/myMap/myMapUN_07.JPG" << ".data/myMap/myMapUN_08.JPG" << ".data/myMap/myMapUN_09.JPG" << ".data/myMap/myMapUN_10.JPG" << ".data/myMap/myMapUN_11.JPG";
        fs << "]";                                           // close sequence*/

        // map
        fs << "mapsNames" << "[";                              // text - string sequence
        fs << "./data/map/01.jpg" << "./data/map/02.jpg" << "./data/map/03.jpg";
        fs << "]";                                            // close sequence

        // map_unwrapped
        fs << "mapsUnNames" << "[";                              // text - string sequence
        fs << "./data/map/01_UN.jpg" << "./data/map/02_UN.jpg" << "./data/map/03_UN.jpg";
        fs << "]";                                            // close sequence*/
        
        fs << "calibrationFile" << "./data/intrinsic_calibration.xml";

        //--------------> mask
        // fs << "blackMask"   << "[" << 0 << 0 << 0 << 179 << 255 << 70 <<     "]"; //black
        // fs << "redMask"     << "[" << 15 << 100 << 140 << 160 << 255 << 255 << "]"; //red
        // // fs << "greenMask"   << "[" << 50 << 65 << 45 << 70 << 255 << 200 <<   "]"; //green
        // fs << "blueMask"    << "[" << 100 << 100 << 40 << 140 << 200 << 170 <<  "]"; //blue
        fs << "blackMask"   << "[" << 0 << 0 << 0 << 179 << 255 << 70 <<     "]"; //black
        fs << "redMask"     << "[" << 10 << 0 << 0 << 170 << 255 << 255 << "]"; //red then bitwise

        fs << "greenMask"   << "[" << 40 << 60 << 80 << 80 << 255 << 170 << "]"; //green
        fs << "blueMask"    << "[" << 100 << 100 << 40 << 140 << 200 << 170 <<  "]"; //blue

        // filtering
        fs << "kernelSide" << 9;
        fs << "convexHullFile" << "data/convexHull.xml";
        fs << "templatesFolder" << "data/num_template/";
        fs << "templates" << "[" << "0.png" << "1.png" << "2.png" << "3.png" << "4.png" << "5.png" << "6.png" << "7.png" << "8.png" << "9.png" << "0_inv.png" << "1_inv.png" << "2_inv.png" << "3_inv.png" << "4_inv.png" << "5_inv.png" << "6_inv.png" << "7_inv.png" << "8_inv.png" << "9_inv.png" << "]";

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