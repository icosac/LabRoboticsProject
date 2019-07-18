#include"planning.hh"

//debug blocks most things, wait only something
// #define DEBUG
// #define WAIT

void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fs){
    /*FileStorage fs(loadFile, FileStorage::READ);

    FileNode data = fs["obstacles"];
    for (FileNodeIterator itData = data.begin(); itData != data.end(); itData++){

        // Read each vector
        vector<Point2<int> > vp;

        FileNode pts = *itData; //points
        for (FileNodeIterator itPts = pts.begin(); itPts != pts.end(); ++itPts){
            // Read each point            
            FileNode pt = *itPts;   // point
            FileNodeIterator itPt = pt.begin(); //point iterator

            int x = *itPt; itPt++;
            int y = *itPt; itPt++;

            vp.push_back(Point2<int>(x, y));
        }
        vvp.push_back(vp);
    }*/

    cout << "size: " << fs.size() << endl; 
    // for(unsigned int i=0; i<fs.size(); i++){
    //     for(unsigned int j=0; j<(); j++){
            
    //     }
    //     cout << endl;
    // }
}

pair< vector<Point2<int> >, Mat > planning(){
    FileStorage fs_xml("data/settings.xml", FileStorage::READ);
    string loadFile = (string)fs_xml["convexHullFile"];
    fs_xml.release();
    cout << "loadFile " << loadFile << endl;

    FileStorage fs(loadFile, FileStorage::READ);
    cout << "ok " << endl;
    vector< vector<Point2<int> > > vvp;
    loadVVP(vvp, fs["obstacles"]);
    cout << "ok " << endl;

    for(unsigned int i=0; i<vvp.size(); i++){
        for(unsigned int j=0; j<vvp[i].size(); j++){
            cout << vvp[i][j] << " ";
        }
        cout << endl;
    }
    cout << "ok " << endl;

}

