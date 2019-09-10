#include"planning.hh"

namespace Planning {
    Mapp* map;

    vector<Point2<int> > new_function(vector<vector<Point2<int> > > vvp){
        vector<Point2<int> > v;
        for (auto a : vvp){
            for (auto b : a){
                v.push_back(b);
            }
        }
        return v;
    }

    /*! \brief The function plan a route from the actual position of the robot up to the final gate through all the victims.
        \details All the data about the objects are loaded from the files previously saved. Then a Mapp is created and on that structure, thanks to a minPath function and a lot of dubin curves, the best route is computed.

        \param[in] img It is a raw image of the scene that will be used from the localize function to find the starting state of the robot.
        \returns Two elements are returned: a pointer to the Mapp where all data are stored and a vector of points placed on the computed route.
    */
    // pair< vector<Point2<int> >, Mapp* > Planning::planning(const Mat & img){
    vector<Point2<int> > planning(const Mat & img){
        cout << "plan0\n" << flush;
        Planning::createMapp();
        cout << "plan1\n" << flush;

        vector<Point2<int> > vp;
        cout << "plan2\n" << flush;

        // use this version when run from the laboratory... 
        Configuration2<double> conf = localize(img, true);
        vp.push_back( Point2<int>( (int)conf.y(), (int)conf.x()) ); //robot initial location. Inverted because inverted before in localize. 
        // 
        // vp.push_back( Point2<int>(900, 300) );//*/
        cout << "plan3\n" << flush;
        Planning::map->getVictimCenters(vp);
        cout << "plan3 bis\n" << flush;
        Planning::map->getGateCenter(vp);
        cout << "plan4\n" << flush;

        #define BEST
        #ifdef BEST 
            vector<vector<Point2<int> > > vvp = map->minPathNPointsWithChoice(vp, 4);
        #else
            vector<vector<Point2<int> > > vvp = map->minPathNPoints(vp);
        #endif
        cout << "plan5\n" << flush;

        //Add Dubins
        // cout << "Trying to use dubins" << endl << flush;
        // Planning::plan_best(conf, vvp);
        // cout << "Found best dubins" << endl;

        vector<Point2<int> > cellsOfPath = new_function(vvp);
        // vector<Point2<int> > cellsOfPath = map->sampleNPoints(vvp);

        cout << "plan6\n" << flush;
        cout << "\tCellsOfPath size: " << cellsOfPath.size() <<endl;
        cout << "plan7\n" << flush;
        Mat imageMap = Planning::map->createMapRepresentation();
        cout << "Created map" << endl << flush;
        Planning::map->imageAddPoints(imageMap, cellsOfPath);
        Planning::map->imageAddSegments(imageMap, cellsOfPath);

        imwrite("data/computePlanning.jpg", imageMap);
        #ifdef WAIT
            namedWindow("Map", WINDOW_NORMAL);
            imshow("Map", imageMap);
            mywaitkey();
        #endif

        delete Planning::map;
        return( cellsOfPath );
    }

    /*! \brief The goal is to load, all the neccessary data, from files and create a Mapp that store everything.

        \returns The created mapp.
    */
    void createMapp(){
        sett->cleanAndRead();
        cout << "create0\n" << flush;

        //create the map
        int dimX=1000, dimY=1500;
        Planning::map = new Mapp(dimX, dimY);
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
            Planning::loadVVP(vvpObstacles, fs["obstacles"]);
            cout << "create2 c\n" << flush;
            vector<Obstacle> obstacles;
            for(unsigned int i=0; i<vvpObstacles.size(); i++){
                obstacles.push_back( Obstacle(vvpObstacles[i]) );
            }
            cout << "create2 d\n" << flush;
            Planning::map->addObjects(obstacles);
            cout << "create2 e\n" << flush;
            if(obstacles.size()==0){
                throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no obstacles for the creating of the map.", __LINE__, __FILE__);
            }
            cout << "create3\n" << flush;

            // Victims
            vector< vector<Point2<int> > > vvpVictims;
            Planning::loadVVP(vvpVictims, fs["victims"]);
            vector<Victim> victims;
            for(unsigned int i=0; i<vvpVictims.size(); i++){
                victims.push_back( Victim(vvpVictims[i], i+1) );    // the victims are already sorted
            }
            Planning::map->addObjects(victims);
            if(victims.size()==0){
                cerr << "Warning: Loaded no victims for the creating of the map. " << __LINE__ << " " << __FILE__ << endl;
            }
            cout << "create4\n" << flush;

            // Gate
            vector< vector<Point2<int> > > vvpGates;
            Planning::loadVVP(vvpGates, fs["gate"]);
            vector<Gate> gates;
            for(unsigned int i=0; i<vvpGates.size(); i++){
                gates.push_back( Gate(vvpGates[i]) );
            }
            Planning::map->addObjects(gates);
            if(gates.size()==0){
                throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no gate for the creating of the map.", __LINE__, __FILE__);
            }
        cout << "create5\n" << flush;

        // return(map);
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
            double th=0.0, s=0.0, k=0.1; // k will be computed thanks to the dubins
            vp[0].invert();
            unsigned int i;
            for(i=0; i<vp.size()-1; i++){
                vp[i+1].invert();
                th = vp[i].th( vp[i+1] ).toRad();

                if (!equal(th, 0.01)){
                    k=0;
                }
                else if (th<0){
                    k=-k;
                }

                pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, th, k);
                path.add(pose);

                s += vp[i].distance(vp[i+1])/SCALE;
            }
            pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, th, k);//use last th and k or simply 0????
            path.add(pose);

            cout << "Path elements:\n";
            for(i=0; i<path.size(); i++){
                cout << "pose " << i+1 << "°: " << path.points[i].string().str() << endl;
            }
        } else{
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Impossible to convert vector to path, dimension less than 2.", __LINE__, __FILE__);
        }
    }


    // TODO should I keep this or not?
    #define DELTA M_PI/180.0 //1 degree

    template <class T>
    vector<Point2<T> > reduce_points(Tuple<Point2<T> > init_points){
      vector<Point2<T> > ret={};
      for (int i=0; i<init_points.size(); i++){
        Point2<T> app=init_points.get(i);
        if (i==0 || i==init_points.size()-1){
          ret.push_back(app);
        }
        else {
          if (ret.back().th(app).toRad()>DELTA){
            ret.push_back(app);
          }
        }
      }
      return ret;
    }

    #define ROB_KMAX 100

    template<class T>
    void start_pos( const Configuration2<double>& _start, 
                    vector<Point2<T> >& vvPoints){
        Dubins<double> start_dub;
        bool ok=false;
        vector<Point2<T> > new_start;
        for (uint i=0; (i<(vvPoints.size()-1)  && start_dub.getId()<0 && !ok); i++){ //I continue until the points are empty or until a feasible dubins is not found. 
            ok=true; //Need to reset this each loop
            new_start.clear();
            start_dub=Dubins<double>(_start, Configuration2<double>(vvPoints[i], vvPoints[i].th(vvPoints[i+1])), ROB_KMAX);
            Tuple<Tuple<Point2<double> > > vPDub=start_dub.splitIt(); 
            for (int j=0; (j<3 && ok); j++){    
                for (int k=0; (k<(vPDub[j].size()-1) && ok); k++){    
                    if (Planning::map->checkSegment(vPDub[j][k], vPDub[j][k+1])){
                        cerr << "Segment through obstacles" << endl;
                        ok=false;
                    }
                    else if(vPDub[j][k].x()<Planning::map->getActualLengthX() && vPDub[j][k].y()<Planning::map->getActualLengthY() 
                            && vPDub[j][k].x()>Planning::map->getOffsetValue() && vPDub[j][k].y()>Planning::map->getOffsetValue()){
                        cerr << "Point out of map " << vPDub[j][k] << endl;
                        ok=false;
                    }
                    else {
                        new_start.push_back(vPDub[j][k]);
                    }
                }
                if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1])!=OBJ_TYPE::OBST){
                    cerr << "Last point is on obstacle" << endl;
                    ok=false;
                }
            }
        }
        if (start_dub.getId()>=0){
            cout << start_dub << endl;
            cout << "NEW START" << endl;
            for (auto a : new_start){
                cout << a << endl;
            }
            vvPoints=new_start;
        }
        else {
            throw MyException<string> (EXCEPTION_TYPE::GENERAL, "No feasible path was found", __LINE__, __FILE__);
        }

    }

    // template<class T>
    void plan_best( const Configuration2<double>& _start,
                    vector<vector<Point2<int> > >& vvPoints){
        start_pos(_start, vvPoints[0]);
    }

}



