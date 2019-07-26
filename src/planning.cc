#include"planning.hh"

/*! \brief The function plan a route from the actual position of the robot up to the final gate through all the victims.
    \details All the data about the objects are loaded from the files previously saved. Then a Mapp is created and on that structure, thanks to a minPath function and a lot of dubin curves, the best route is computed.

    \returns Two elements are returned: a pointer to the Mapp where all data are stored and a vector of points placed on the computed route.
*/
pair< vector<Point2<int> >, Mapp* > planning(){
    Mapp * map = createMapp();

    // TODO test and verify
    vector<Point2<int> > vp;
    vp.push_back( Point2<int>(100, 150) );
    vp.push_back( Point2<int>(200, 300) );
    vp.push_back( Point2<int>(500, 750) );
    vp.push_back( Point2<int>(50, 1100) );
    vp.push_back( Point2<int>(900, 1450));
    vp.push_back( Point2<int>(800, 200) );
    vp.push_back( Point2<int>(300, 100) );

    vector<Point2<int> > * cellsOfPath = new vector<Point2<int> >();
    if(true){
        cellsOfPath = map->minPathTwoPoints(vp[3], vp[4]);
    } else {
        vector<vector<Point2<int> > > * vvp = map->minPathNPoints(vp);
        for(unsigned int i=0; i<vvp->size(); i++){
            for(unsigned int j=0; j<vvp[i].size(); j++){
                // cellsOfPath->push_back();
                //cout << vvp[i][j] << endl;
            }
        }
    }
    cout << "cellsOfPath size: " << cellsOfPath->size() <<endl;
    
    return( make_pair(*cellsOfPath, map) );//todo change with points from dubins
}

/*! \brief The goal is to load, all the neccessary data, from files and create a Mapp that store everything.

    \returns The created mapp.
*/
Mapp * createMapp(){
    sett->cleanAndRead();

    // open file
    cout << "loadFile: " << sett->convexHullFile << endl;
    FileStorage fs(sett->convexHullFile, FileStorage::READ);
    
    // load vectors of vectors of objects
    vector< vector<Point2<int> > > obstacles;
    loadVVP(obstacles, fs["obstacles"]);

    vector< vector<Point2<int> > > victims;
    loadVVP(victims, fs["victims"]);

    vector< vector<Point2<int> > > gate;
    loadVVP(gate, fs["gate"]);

    //create the map
    cout << "MAIN MAP\n";
    int dimX=1000, dimY=1500;
    Mapp * map = new Mapp(dimX, dimY, 5, 5);
    
    map->addObjects(obstacles, OBST);
    map->addObjects(victims, VICT);
    map->addObjects(gate, GATE);

    return(map);
}

/*! \brief The function load from the given fileNode a vector of vectors of Point2<int>.

    \param[out] vvp The location where to save the loaded vector of vectors.
    \param[in] fn The fileNode from which to load the vector of vectors.
*/
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

/*! \brief The function load from the given fileNode a vector of Point2<int>.

    \param[out] vp The location where to save the loaded vector.
    \param[in] fn The fileNode from which to load the vector.
*/
void loadVP(vector<Point2<int> > & vp, FileNode fn){
    FileNode data = fn; //points
    for (FileNodeIterator itPts = data.begin(); itPts != data.end(); itPts++){
        int x = *itPts; 
        itPts++;
        int y = *itPts;
        vp.push_back(Point2<int>(x, y));
    }
}
