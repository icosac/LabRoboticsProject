#include"planning.hh"

//debug blocks most things, wait only something
// #define DEBUG
// #define WAIT

pair< vector<Point2<int> >, Mat > planning(){
//void planning(){
    // load file of saving
    FileStorage fs_xml("data/settings.xml", FileStorage::READ);
    string loadFile = (string)fs_xml["convexHullFile"];
    fs_xml.release();

    // open file
    cout << "loadFile: " << loadFile << endl;
    FileStorage fs(loadFile, FileStorage::READ);
    
    // load vectors of objects
    vector< vector<Point2<int> > > obstacles;
    loadVVP(obstacles, fs["obstacles"]);

    vector< vector<Point2<int> > > victims;
    loadVVP(victims, fs["victims"]);

    vector< vector<Point2<int> > > gate;
    loadVVP(gate, fs["gate"]);

    //create the map
    cout << "MAIN MAP\n";
    int dimX=1000, dimY=1500;
    Mapp* map = new Mapp(dimX, dimY, 5, 5);
    
    map->addObjects(obstacles, OBST);
    map->addObjects(victims, VICT);
    map->addObjects(gate, GATE);
    

    // Generate the map representation and print it
    Mat imageMap = map->createMapRepresentation();

    // TODO test and verify
    Point2<int> start(120, 120);
    Point2<int> end(900, 1400);
    vector<Point2<int> > cellsOfPath = map->minPathTwoPoints(start, end);
    cout << "cellsOfPath size: " << cellsOfPath.size() <<endl;
    
    return( make_pair(cellsOfPath, imageMap) );/*todo change with points from dubins*/
}

void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn){
    FileNode data = fn;
    for (FileNodeIterator itData = data.begin(); itData != data.end(); itData++){

        // Read the vector
        vector<Point2<int> > vp;

        FileNode pts = *itData; //points
        for (FileNodeIterator itPts = pts.begin(); itPts != pts.end(); itPts++){
            int x = *itPts; 
            itPts++;
            int y = *itPts;

            vp.push_back(Point2<int>(x, y));
        }
        vvp.push_back(vp);
    }
}

void loadVP(vector<Point2<int> > & vp, FileNode fn){
    // useless function at the moment
    FileNode data = fn; //points
    for (FileNodeIterator itPts = data.begin(); itPts != data.end(); itPts++){
        int x = *itPts; 
        itPts++;
        int y = *itPts;
        vp.push_back(Point2<int>(x, y));
    }
}
