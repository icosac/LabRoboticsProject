#include"planning.hh"

//debug blocks most things, wait only something
// #define DEBUG
// #define WAIT

//pair< vector<Point2<int> >, Mat > planning(){
void planning(){
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

    vector<Point2<int> > gate;
    loadVP(gate, fs["gate"]);

    //create the map
    cout << "MAIN MAP\n";
    int dimX=1000, dimY=1500;
    cout << "a\n"; Mapp* map = new Mapp(dimX, dimY, 5, 5);
    
    cout << "b\n"; map->addObjects(obstacles, OBST);
    cout << "c\n"; map->addObjects(victims, VICT);
    cout << "d\n"; 
    for(unsigned int i=0; i<gate.size(); i++){
        cout << gate[i] << " --- ";
    }
    cout << endl << endl;
    map->addObject(gate, GATE);
    cout << "e\n"; 

    // Generate the map representation and print it
    // map->printMap();
    // Mat imageMap = map->createMapRepresentation();
    // namedWindow("Map", WINDOW_AUTOSIZE);
	// imshow("Map", imageMap);

    // Point2<int> start(50, 70);
    // Point2<int> end(80, 20);
    // vector<Point2<int> > cellsOfPath = map->minPathTwoPoints(start, end);
    
    // vector< vector<Point2<int> > > dubins;
    // return( make_pair(gate, imageMap) );/*todo change with dubins*/
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
    FileNode pts = fn; //points
    for (FileNodeIterator itPts = pts.begin(); itPts != pts.end(); itPts++){
        int x = *itPts; 
        itPts++;
        int y = *itPts;

        vp.push_back(Point2<int>(x, y));
    }
}
