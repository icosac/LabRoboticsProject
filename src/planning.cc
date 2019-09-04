#include"planning.hh"

/*! \brief The function plan a route from the actual position of the robot up to the final gate through all the victims.
    \details All the data about the objects are loaded from the files previously saved. Then a Mapp is created and on that structure, thanks to a minPath function and a lot of dubin curves, the best route is computed.

    \returns Two elements are returned: a pointer to the Mapp where all data are stored and a vector of points placed on the computed route.
*/
pair< vector<Point2<int> >, Mapp* > planning(){
    Mapp * map = createMapp();

    vector<Point2<int> > vp;

    // TODO use this version when run from the laboratory...
    //vp.push_back( localize() ); //robot initial location
    vp.push_back( Point2<int>(100, 150) );
    
    map->getVictimCenters(vp);
    map->getGateCenter(vp);

    vector<vector<Point2<int> > > vvp = map->minPathNPoints(vp);
    vector<Point2<int> > cellsOfPath = map->sampleNPoints(vvp);

    cout << "\tCellsOfPath size: " << cellsOfPath.size() <<endl;

    return( make_pair(cellsOfPath, map) );//todo change with points from dubins
}

/*! \brief The goal is to load, all the neccessary data, from files and create a Mapp that store everything.

    \returns The created mapp.
*/
Mapp * createMapp(){
    sett->cleanAndRead();

    //create the map
    int dimX=1000, dimY=1500;
    Mapp * map = new Mapp(dimX, dimY);

    // open file
    #ifdef WAIT
        cout << "loadFile: " << sett->convexHullFile << endl;
    #endif
    FileStorage fs(sett->convexHullFile, FileStorage::READ);
    
    // load vectors of vectors of objects
        // Obstacles
        vector< vector<Point2<int> > > vvpObstacles;
        loadVVP(vvpObstacles, fs["obstacles"]);
        vector<Obstacle> obstacles;
        for(unsigned int i=0; i<vvpObstacles.size(); i++){
            obstacles.push_back( Obstacle(vvpObstacles[i]) );
        }
        map->addObjects(obstacles);

        // Victims
        vector< vector<Point2<int> > > vvpVictims;
        loadVVP(vvpVictims, fs["victims"]);
        vector<Victim> victims;
        for(unsigned int i=0; i<vvpVictims.size(); i++){
            victims.push_back( Victim(vvpVictims[i], i+1) );    // the victims are already sorted
        }
        map->addObjects(victims);

        // Gate
        vector< vector<Point2<int> > > vvpGates;
        loadVVP(vvpGates, fs["gate"]);
        vector<Gate> gates;
        for(unsigned int i=0; i<vvpGates.size(); i++){
            gates.push_back( Gate(vvpGates[i]) );
        }
        map->addObjects(gates);

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
