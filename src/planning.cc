#include"planning.hh"

/*! \brief The function plan a route from the actual position of the robot up to the final gate through all the victims.
    \details All the data about the objects are loaded from the files previously saved. Then a Mapp is created and on that structure, thanks to a minPath function and a lot of dubin curves, the best route is computed.

    \param[in] img It is a raw image of the scene that will be used from the localize function to find the starting state of the robot.
    \returns Two elements are returned: a pointer to the Mapp where all data are stored and a vector of points placed on the computed route.
*/
pair< vector<Point2<int> >, Mapp* > planning(const Mat & img){
    cout << "plan0\n" << flush;
    Mapp * map = createMapp();
    cout << "plan1\n" << flush;

    vector<Point2<int> > vp;
    cout << "plan2\n" << flush;

    // use this version when run from the laboratory... 
    Configuration2<double> conf = localize(img, true);
    vp.push_back( Point2<int>( (int)conf.x(), (int)conf.y()) ); //robot initial location
    /*/ 
    vp.push_back( Point2<int>(100, 150) );//*/
    cout << "plan3\n" << flush;
    map->getVictimCenters(vp);
    cout << "plan3 bis\n" << flush;
    map->getGateCenter(vp);
    cout << "plan4\n" << flush;

    vector<vector<Point2<int> > > vvp = map->minPathNPoints(vp);
    cout << "plan5\n" << flush;
    vector<Point2<int> > cellsOfPath = map->sampleNPoints(vvp, 50);
    cout << "plan6\n" << flush;

    cout << "\tCellsOfPath size: " << cellsOfPath.size() <<endl;
    cout << "plan7\n" << flush;

    return( make_pair(cellsOfPath, map) );
    // return( make_pair( vector<Point2<int> >(), map) );
}

/*! \brief The goal is to load, all the neccessary data, from files and create a Mapp that store everything.

    \returns The created mapp.
*/
Mapp * createMapp(){
    sett->cleanAndRead();
    cout << "create0\n" << flush;

    //create the map
    int dimX=1000, dimY=1500;
    Mapp * map = new Mapp(dimX, dimY);
    cout << "create1\n" << flush;

    // open file
    #ifdef WAIT
        cout << "loadFile: " << sett->convexHullFile << endl;
    #endif
    FileStorage fs(sett->convexHullFile, FileStorage::READ);
    cout << "create2 a\n" << flush;
    
    // load vectors of vectors of objects
        // Obstacles
        vector< vector<Point2<int> > > vvpObstacles;
        cout << "create2 b\n" << flush;
        loadVVP(vvpObstacles, fs["obstacles"]);
        cout << "create2 c\n" << flush;
        vector<Obstacle> obstacles;
        for(unsigned int i=0; i<vvpObstacles.size(); i++){
            obstacles.push_back( Obstacle(vvpObstacles[i]) );
        }
        cout << "create2 d\n" << flush;
        map->addObjects(obstacles);
        cout << "create2 e\n" << flush;
        if(obstacles.size()==0){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no obstacles for the creating of the map.", __LINE__, __FILE__);
        }
    cout << "create3\n" << flush;

        // Victims
        vector< vector<Point2<int> > > vvpVictims;
        loadVVP(vvpVictims, fs["victims"]);
        vector<Victim> victims;
        for(unsigned int i=0; i<vvpVictims.size(); i++){
            victims.push_back( Victim(vvpVictims[i], i+1) );    // the victims are already sorted
        }
        map->addObjects(victims);
        if(victims.size()==0){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no victims for the creating of the map.", __LINE__, __FILE__);
        }
        cout << "create4\n" << flush;

        // Gate
        vector< vector<Point2<int> > > vvpGates;
        loadVVP(vvpGates, fs["gate"]);
        vector<Gate> gates;
        for(unsigned int i=0; i<vvpGates.size(); i++){
            gates.push_back( Gate(vvpGates[i]) );
        }
        map->addObjects(gates);
        if(gates.size()==0){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no gate for the creating of the map.", __LINE__, __FILE__);
        }
    cout << "create5\n" << flush;

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

#define SCALE 1000.0
/*! \brief Convert a vector of point to a path, from Enrico's notation to Paolo's notation.

    \param[in] vp The sorce vector.
    \param[out] path The destination path.
*/
void fromVpToPath(vector<Point2<int> > & vp, Path & path){
    if(vp.size()>=2){
        path.resize(0);
        Pose pose;
        double th=0.0, s=0.0, k=1; // k will be computed thanks to the dubins
        unsigned int i;
        for(i=0; i<vp.size()-1; i++){
            th = vp[i].th( vp[i+1] ).toRad();

            pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, th, k);
            path.add(pose);

            s += vp[i].distance(vp[i+1])/SCALE;
        }
        pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, th, k);//use last th and k or simply 0????
        path.add(pose);

        cout << "Path elements:\n";
        for(Pose p : path.points){
            cout << p.string().str() << endl;
        }
    } else{
        throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Impossible to convert vector to path, dimension less than 2.", __LINE__, __FILE__);
    }
}