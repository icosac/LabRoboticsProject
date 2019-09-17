#include"planning.hh"

namespace Planning {
    Mapp* map;
    Configuration2<double> conf;
    const double angleRange = 12*M_PI/180;
    const int nAngles = 90;
    const int range = 3;   

    // TODO should I keep this or not?
    #define DELTA M_PI/180.0 //1 degree
    #define ROB_KMAX KMAX //0.01
    #define ROB_PIECE_LENGTH 20
    #define BONUS 5

    /*! \brief Convert a vector of vector of points into a vector of points (AKA collapse everything).

        \param[in] The vector of vector of points that needs to be collapsed
        \returns The new vector of points.
    */
    vector<Point2<int> > convertToVP(const vector<vector<Point2<int> > > & arr){
        vector<Point2<int> > v;
        for (auto a : arr){
            for (auto b : a){
                v.push_back(b);
            }
        }
        return v;
    }

    /*! \brief Convert a vector of vector of configurations into a vector of points (AKA collapse everything).

        \param[in] The vector of vector of configurations that needs to be collapsed
        \returns The new vector of points.
    */
    vector<Point2<int> > convertToVP(const vector<vector<Configuration2<double> > > & arr){
        vector<Point2<int> > v;
        for (auto a : arr){
            for (auto b : a){
                v.push_back(b.point());
            }
        }
        return v;
    }

    /*! \brief Convert a vector of vector of configurations into a vector of configurations (AKA collapse everything).

        \param[in] The vector of vector of configurations that needs to be collapsed
        \returns The new vector of configurations.
    */
    vector<Configuration2<double> > convertToVC(const vector<vector<Configuration2<double> > > & arr){
        vector<Configuration2<double> > v;
        for (auto a : arr){
            for (auto b : a){
                v.push_back(b);
            }
        }
        return v;
    }

    /*! \brief Convert a vector of vector of points into a vector of configurations (AKA collapse everything).

        \param[in] The vector of vector of points that needs to be collapsed
        \returns The new vector of configurations.
    */
    vector<Configuration2<double> > convertToVC(const vector<vector<Point2<int> > > & arr){
        vector<Point2<int> > v = convertToVP(arr);
        vector<Configuration2<double> > ret;
        uint i=0;
        for(; i<v.size()-1; i++){
            ret.push_back(Configuration2<double>( (double)v[i].x(), (double)v[i].y(), v[i].th(v[i+1]) ));
        }
        ret.push_back(Configuration2<double>( (double)v[i].x(), (double)v[i].y(), v[i-1].th(v[i]) ));
        return ret;
    }

    /*! \brief Show in a window the representation of the map with the addition of the points and segment taken from the parameters.

        \param[in] vv A vector of vector of points that will be added to the map.
        \param[in] name The name of the window that will be created.
    */
    void draw(const vector<vector<Point2<int> > > & vv, string name){
        #if defined WAIT || defined SHOW_MAP
            Mat imageMap = Planning::map->createMapRepresentation();
            vector<Point2<int> > cellsOfPath = convertToVP(vv);
            Planning::map->imageAddPoints(imageMap, cellsOfPath);
            Planning::map->imageAddSegments(imageMap, cellsOfPath);
            
            namedWindow(name.c_str(), WINDOW_NORMAL);
            imshow(name.c_str(), imageMap);
            mywaitkey();
        #endif
    }

    /*! \brief Show in a window the representation of the map with the addition of the configurations and segment taken from the parameters.

        \param[in] vv A vector of vector of configurations that will be added to the map.
        \param[in] name The name of the window that will be created.
    */
    void draw(const vector<vector<Configuration2<double> > > & vv, string name){
        #if defined WAIT || defined SHOW_MAP
            Mat imageMap = Planning::map->createMapRepresentation();
            vector<Point2<int> > cellsOfPath = convertToVP(vv);
            Planning::map->imageAddPoints(imageMap, cellsOfPath);
            Planning::map->imageAddSegments(imageMap, cellsOfPath);
            
            namedWindow(name.c_str(), WINDOW_NORMAL);
            imshow(name.c_str(), imageMap);
            mywaitkey();
        #endif
    }

    /*! \brief Show in a window the representation of the map with the addition of the configurations and segment taken from the parameters. Plus a set of grey points (left vector) and black points (right vector).

        \param[in] vv A vector of vector of configurations that will be added to the map.
        \param[in] left A set of grey points will be added to the map.
        \param[in] right A set of bleck points will be added to the map.
        \param[in] name The name of the window that will be created.
    */
    void draw(  const vector<vector<Configuration2<double> > > & vv, 
                const vector<Configuration2<double> > & left, 
                const vector<Configuration2<double> > & right, 
                string name)
    {
        #if defined WAIT || defined SHOW_MAP
            Mat imageMap = Planning::map->createMapRepresentation();
            vector<Point2<int> > cellsOfPath = convertToVP(vv);
            Planning::map->imageAddPoints(imageMap, cellsOfPath);
            Planning::map->imageAddSegments(imageMap, cellsOfPath);
            
            for (auto a : left){
                Planning::map->imageAddPoint(imageMap, a.point(), 9, Scalar(100, 100, 100));
            }

            for (auto a : right){
                Planning::map->imageAddPoint(imageMap, a.point(), 9, Scalar(0, 0, 0));
            }
            

            namedWindow(name.c_str(), WINDOW_NORMAL);
            imshow(name.c_str(), imageMap);
            mywaitkey();
        #endif
    }

    /*! \brief The function plan a route from the actual position of the robot up to the final gate through all the victims.
        \details All the data about the objects are loaded from the files previously saved. Then a Mapp is created and on that structure, thanks to a minPath function and a lot of dubin curves, the best route is computed.

        \param[in] img It is a raw image of the scene that will be used from the localize function to find the starting state of the robot.
        \returns Two elements are returned: a pointer to the Mapp where all data are stored and a vector of points placed on the computed route.
    */
    vector<Configuration2<double> > planning(const Mat & img){
        Planning::createMapp();

        // localize the robot
        Configuration2<double> conf = ::localize(img, true);
        vector<Point2<int> > vp;
        vp.push_back( Point2<int>( (int)conf.x(), (int)conf.y()) ); //robot initial location.

        // update and show the created map
        Planning::map->getVictimCenters(vp);
        Planning::map->getGateCenter(vp);


        Mat imageMap = Planning::map->createMapRepresentation();
        Planning::map->imageAddPoint(imageMap, conf.point(), 10, Scalar(0,0,0));
        Planning::map->imageAddPoint(imageMap, conf.point().offset(20, conf.angle()), 10, Scalar(100,100,100));

        cout << "Created map" << endl << flush;

        #if defined WAIT || defined SHOW_MAP
            namedWindow("Map 0", WINDOW_NORMAL);
            imshow("Map 0", imageMap);
            mywaitkey();
        #endif

        // search for the shortest path
        // #define BEST
        #ifdef BEST 
            vector<vector<Point2<int> > > vvp = minPathNPointsWithChoice(vp, BONUS, false);
        #else
            vector<vector<Point2<int> > > vvp = minPathNPoints(vp, false);
        #endif
        
        cout << "Path planned" << endl;

        vector<Configuration2<double> > cellsOfPath = convertToVC(vvp);
        Planning::map->imageAddPoints(imageMap, cellsOfPath);
        Planning::map->imageAddSegments(imageMap, cellsOfPath);

        #if defined WAIT || defined SHOW_MAP
            namedWindow("Map2", WINDOW_NORMAL);
            imshow("Map2", imageMap);
            mywaitkey();
        #endif
        cout << "created second map" << endl;
        // #define NOT_DUBINS 
        #ifndef NOT_DUBINS
            //Add Dubins
            cout << "Trying to use dubins" << endl << flush;
            vector<vector<Configuration2<double> > > vvc(vvp.size());
            for (uint i=0; i<vvp.size(); i++){
                for (uint j=0; j<vvp[i].size(); j++){
                    if (j==(vvp[i].size()-1)){
                        vvc[i].push_back(Configuration2<double> (Point2<double>((double)vvp[i][j].x(), (double)vvp[i][j].y()), vvp[i][j-1].th(vvp[i][j])));
                    }
                    else {
                        vvc[i].push_back(Configuration2<double> (Point2<double>((double)vvp[i][j].x(), (double)vvp[i][j].y()), vvp[i][j].th(vvp[i][j+1])));
                    }
                }
            }

            cout << "Calling dubins function" << endl;
            Planning::plan_dubins(conf, vvc);
            cout << "Found best dubins" << endl;


            cellsOfPath = convertToVC(vvc);
            cout << "\tCellsOfPath size: " << cellsOfPath.size() <<endl;
        #endif

        Mat imageMap2 = Planning::map->createMapRepresentation();
        Planning::map->imageAddPoints(imageMap2, cellsOfPath);
        Planning::map->imageAddSegments(imageMap2, cellsOfPath);

        imwrite("data/computePlanning.jpg", imageMap2);
        #if defined WAIT || defined SHOW_MAP
            namedWindow("Map Final", WINDOW_NORMAL);
            imshow("Map Final", imageMap2);
            mywaitkey();
        #endif

        return( cellsOfPath );
    }

    /*! \brief The goal is to load, all the neccessary data, from files and create a Mapp that store everything.

        \returns The created mapp.
    */
    void createMapp(){
        sett->cleanAndRead();

        //create the map
        int dimX=1000, dimY=1500;
        Planning::map = new Mapp(dimX, dimY);

        // open file
        #ifdef WAIT
            cout << "loadFile: " << sett->convexHullFile << endl;
        #endif
        FileStorage fs(sett->convexHullFile, FileStorage::READ);
        
        // load vectors of vectors of objects

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

        // Gate
        vector< vector<Point2<int> > > vvpGates;
        Planning::loadVVP(vvpGates, fs["gate"]);

        vector<Gate> gates;
        for(unsigned int i=0; i<vvpGates.size(); i++){
            gates.push_back( Gate(vvpGates[i]) );

            for(auto el : vvpGates[i]){
                cout << el << " ";
            }
            cout << endl;
            gates[i].print();
        }
        Planning::map->addObjects(gates);
        if(gates.size()==0){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Loaded no gate for the creating of the map.", __LINE__, __FILE__);
        }

        // Obstacles
        vector< vector<Point2<int> > > vvpObstacles;
        Planning::loadVVP(vvpObstacles, fs["obstacles"]);
        vector<Obstacle> obstacles;
        for(unsigned int i=0; i<vvpObstacles.size(); i++){
            obstacles.push_back( Obstacle(vvpObstacles[i]) );
        }
        Planning::map->addObjects(obstacles);
        if(obstacles.size()==0){
            cerr << "Warning: Loaded no obstacles for the creating of the map. " << __LINE__ << " " << __FILE__ << endl;
        }

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

    /*! \brief Get the numper of points needed for the function sampleNpoints.
        \returns The number of points.
    */
    int getNPoints() { return nPoints; }

    /*! \brief Allocate a dynamic 2D array of int.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \returns The allocated array.
    */
    int ** allocateAAInt(const int a, const int b){
        int ** arr = new int*[a];
        for(int i=0; i<a; i++){
            arr[i] = new int[b];
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 3D array of int.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \param[in] c The third dimension.
        \returns The allocated array.
    */
    int *** allocateAAAInt(const int a, const int b, const int c){
        int *** arr = new int**[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAInt(b, c);
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 4D array of int.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \param[in] c The third dimension.
        \param[in] d The fourth dimension.
        \returns The allocated array.
    */
    int **** allocateAAAAInt(const int a, const int b, const int c, const int d){
        int **** arr = new int***[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAAInt(b, c, d);
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 2D array of int.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \returns The allocated array.
    */
    double ** allocateAADouble(const int a, const int b){
        double ** arr = new double*[a];
        for(int i=0; i<a; i++){
            arr[i] = new double[b];
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 3D array of double.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \param[in] c The third dimension.
        \returns The allocated array.
    */
    double *** allocateAAADouble(const int a, const int b, const int c){
        double *** arr = new double**[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAADouble(b, c);
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 4D array of double.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \param[in] c The third dimension.
        \param[in] d The fourth dimension.
        \returns The allocated array.
    */
    double **** allocateAAAADouble(const int a, const int b, const int c, const int d){
        double **** arr = new double***[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAADouble(b, c, d);
        }
        return(arr);
    }

    /*! \brief Allocate a dynamic 2D array of Points.

        \param[in] a The first dimension.
        \param[in] b The second dimension.
        \returns The allocated array.
    */
    Point2<int> ** allocateAAPointInt(const int a, const int b){
        Point2<int> ** arr = new Point2<int>*[a];
        for(int i=0; i<a; i++){
            arr[i] = new Point2<int>[b];
        }
        return(arr);
    }

    /*! \brief Given couples of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
        \details The function is based on a Breadth-first search (BFS). In addittion, the function considered the bonus choose if it is convinient to collect all the victims or only some of them, the bonus is given for each saved victim.

        \param[in] vp The n points that need to be connected.
        \param[in] bonus It is the time in second as reward for each victim saved.
        \param[in] angle It is a boolean flag that says if (for the segment starting from the robot) call the angle version of the minPath or not.
        \returns A vector of vector of points along the path (one for each cell of the grid of the map). Each vector is the best path for one connection, given n points there are n-1 connecctions.
    */
    vector<vector<Point2<int> > > minPathNPointsWithChoice(const vector<Point2<int> > & vp, const double bonus, const bool angle){
        cout << "Begin minpath with BEST." << endl;
        //allocate
        double ** distances = allocateAADouble(map->getDimY(), map->getDimX());
        Point2<int> ** parents = allocateAAPointInt(map->getDimY(), map->getDimX());
        
        double *** distancesAngles = allocateAAADouble(map->getDimY(), map->getDimX(), nAngles);
        int **** parentsAngles = allocateAAAAInt(map->getDimY(), map->getDimX(), nAngles, 3);

        //compute the simple minPath between all pairs of targets
        int n = vp.size();
        vector< vector< vector<Point2<int> > > > vvvp(n, vector< vector<Point2<int> > >(n));
        vector< vector<int> > vvDist(n, vector<int>(n));

        cout << "\nTable of distances:\n";
        for(int i=0; i<n; i++){
            cout << i << "\t";
        }

        for(int i=0; i<n-1; i++){
            cout << endl << i << ":\t" << flush;
            for(int j=0; j<i; j++) cout << "\t";
            for(int j=i+1; j<n; j++){
                //compute the minPath
                if(angle && i==0){
                    vvvp[i][j] = minPathTwoPointsInternalAngles(vp[i], vp[j], distancesAngles, parentsAngles, conf.angle().toRad());
                } else{
                    vvvp[i][j] = minPathTwoPointsInternal(vp[i], vp[j], distances, parents);
                }

                //compute the overall distance along the computed path and store it
                double dist = 0.0;
                for(int q=0; q<(int)(vvvp[i][j].size())-1; q++){
                    dist += vvvp[i][j][q].distance( vvvp[i][j][q+1] );
                }
                cout << (int)dist << "\t" << flush;
                vvDist[i][j] = dist;
                vvDist[j][i] = dist;
            }
        }
        cout << endl;

        //generate all the permutations (disposizioni = with order) without repetition
        vector<int> targets;
        for(int i=1; i<n-1; i++){
            targets.push_back(i);
        }

        set<int> disposizioni;
        cout << "\nThe possible permutations:\n";
        disposizioni.insert(0); //option for taking no victim
        do {
            for(unsigned int i=0, c=0; i<targets.size(); i++){
                c *= 10;
                c += targets[i];
                disposizioni.insert(c);
            }
        } while ( next_permutation(targets.begin(), targets.end()) );

        //compare all the possibilities given by the permutations
        double gain = bonus*100.0; // the measure unit of the bonus is seconds and the gain is in mm. The scale 100 is given by the speed of the robot: 10cm/s.

        double cost, bestCost = 1000000.0;
        int bestDisp = 0;
        for(int el : disposizioni){
            intToVect(el, targets); //convert back to vector

            cout << el << " ->\tThe cost is: ";
            if(targets.size()==0){
                cost =       vvDist[0][ n-1 ];                            
                cout << (int)vvDist[0][ n-1 ] << " = ";
            } else{
                //calculate the cost (lenght) of the actual path
                cost =       vvDist[0][ targets.front() ];                
                cout << (int)vvDist[0][ targets.front() ] << " + ";
                for(unsigned int q=0; q<targets.size()-1; q++){
                    cost +=      vvDist[ targets[q] ][ targets[q+1] ];   
                    cout << (int)vvDist[ targets[q] ][ targets[q+1] ] << " + ";
                }
                cost +=      vvDist[ targets.back() ][ n-1 ];            
                cout << (int)vvDist[ targets.back() ][ n-1 ] << " - ";
                
                cost -=      gain*targets.size();                        
                cout << (int)gain*targets.size() << " = ";
            }
            cout << cost << endl;

            if(cost<bestCost){
                bestCost = cost;
                bestDisp = el;
            }
        }
        cout << "\nThe best cost is: " << bestCost << ", generated by the disp: " << bestDisp << endl << endl;
        
        intToVect(bestDisp, targets);
        //prepare the return values
        vector< vector<Point2<int> > > vvp;
        if(targets.size()==0){
            vvp.push_back( vvvp[0][ n-1 ]);
        } else{
            vvp.push_back( vvvp[0][targets.front()] );
            for(int q=0; q<(int)(targets.size())-1; q++){
                if(targets[q] < targets[q+1]){
                    vvp.push_back(vvvp[ targets[q] ][ targets[q+1] ]);
                } else{
                    vector<Point2<int> > app = vvvp[ targets[q+1] ][ targets[q] ];
                    reverse(app.begin(), app.end());
                    vvp.push_back(app);
                }
            }
            vvp.push_back( vvvp[ targets.back() ][ n-1 ] );
        }

        //delete
        deleteAA(distances, map->getDimY());
        deleteAA(parents, map->getDimY());

        deleteAAA(distancesAngles, map->getDimY(), map->getDimX());
        deleteAAAA(parentsAngles, map->getDimY(), map->getDimX(), nAngles);
        cout << "End minpath with BEST." << endl;

        return(vvp);
    }

    /*! \brief Given couples of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
        \details The function is based on a Breadth-first search (BFS).

        \param[in] p0 The source point.
        \param[in] p1 The destination point.
        \param[in] angle It is a boolean flag that says if (for the first segment) call the angle version of the minPath or not.
        \returns A vector of vector of points along the path (one for each cell of the grid of the map). Each vector is the best path for one connection, given n points there are n-1 connecctions.
    */
    vector<vector<Point2<int> > > minPathNPoints(const vector<Point2<int> > & vp, const bool angle){
        cout << "Begin minpath." << endl;
        //function
        vector<vector<Point2<int> > > vvp;
        // with Angles version
        int i=0;
        if(angle){
            //allocate
            double *** distances = allocateAAADouble(map->getDimY(), map->getDimX(), nAngles);
            int **** parents = allocateAAAAInt(map->getDimY(), map->getDimX(), nAngles, 3);

            vvp.push_back(  minPathTwoPointsInternalAngles(vp[0], vp[1], distances, parents, conf.angle().toRad() ));

            //delete
            deleteAAA(distances, map->getDimY(), map->getDimX());
            deleteAAAA(parents, map->getDimY(), map->getDimX(), nAngles);
            i++;
        }

        // Normal version
        //allocate
        double ** distances = allocateAADouble(map->getDimY(), map->getDimX());
        Point2<int> ** parents = allocateAAPointInt(map->getDimY(), map->getDimX());

        cout << "vp: " << vp.size() << endl;
        // function
        for(; i<(int)(vp.size())-1; i++){
            cout << "computing " << i << endl;
            vvp.push_back( minPathTwoPointsInternal(vp[i], vp[i+1], distances, parents));
            cout << "end computing " << i << endl;
        }

        //delete
        deleteAA(distances, map->getDimY());
        deleteAA(parents, map->getDimY());
        cout << "End minpath." << endl;
        return(vvp);
    }

    /*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
        \details The function is based on a Breadth-first search (BFS).

        \param[in] p0 The source point.
        \param[in] p1 The destination point.
        \param[in] angle It is a boolean flag that says if call the angle version of the minPath or not.
        \returns A vector of points along the path (one for each cell of the grid of the map).
    */
    vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1, const bool angle){
        vector<Point2<int> > vp;
        if(angle){
            //allocate
            double *** distances = allocateAAADouble(map->getDimY(), map->getDimX(), nAngles);
            int **** parents = allocateAAAAInt(map->getDimY(), map->getDimX(), nAngles, 3);

            vp = minPathTwoPointsInternalAngles(p0, p1, distances, parents, conf.angle().toRad() );

            //delete
            deleteAAA(distances, map->getDimY(), map->getDimX());
            deleteAAAA(parents, map->getDimY(), map->getDimX(), nAngles);

        } else{
            //allocate
            double ** distances = allocateAADouble(map->getDimY(), map->getDimX());
            Point2<int> ** parents = allocateAAPointInt(map->getDimY(), map->getDimX());
            
            vp = minPathTwoPointsInternal(p0, p1, distances, parents);
            
            //delete
            deleteAA(distances, map->getDimY());
            deleteAA(parents, map->getDimY());
        }

        return(vp);
    }

    /*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
        \details The function is based on a Breadth-first search (BFS).

        \param[in] startP The source point.
        \param[in] endP The destination point.
        \param[in] distances A matrix that is needed to store the distances of the visited cells.
        \param[in] parents A matrix that is needed to store the parent of each cell (AKA the one that have discovered that cell with the minimum distance).
        \returns A vector of points along the path (one for each cell of the grid of the map).
    */
    vector<Point2<int> > minPathTwoPointsInternal(
                            const Point2<int> & startP, const Point2<int> & endP, 
                            double ** distances, Point2<int> ** parents)
    {
        resetDistanceMap(distances);

        // P=point, C=cell
        Point2<int> startC(startP.x()/map->getPixX(), startP.y()/map->getPixY()), endC(endP.x()/map->getPixX(), endP.y()/map->getPixY());
        queue<Configuration2<int> > toProcess;
        
        // initialization of BFS
        toProcess.push(Configuration2<int>(startC, 0.0));
        if(map->getCellType(startC.y(), startC.x()) == OBST){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "The start position of the robot is inside an Obstacle!", __LINE__, __FILE__);
        }

        distances[startC.y()/*i=y()*/][startC.x()/*j=x()*/] = 0.0;
        parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/] = startC;
        int found = 0;

        // precompute the computation of the distances and the angle in the square of edges around the cell of interest
        const int r = range; //range from class variable (default=3)
        const int side = 2*r+1;
        double computedDistances[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
        
        for(int i=r; i>=(-r); i--){
            for(int j=(-r); j<=r; j++){
                computedDistances[(i+r)*side + (j+r)]  = sqrt( pow(i,2) + pow(j,2) );
            }
        }
        cout << "start iteration" << endl;
        // start iteration of the BFS
        while( !toProcess.empty() && found<=foundLimit ){
            // for each cell from the queue
            Configuration2<int> cell = toProcess.front();
            toProcess.pop();

            int iC = (int)cell.y(), jC = (int)cell.x(); //i and j of the cell
            double dist = distances[iC][jC];
                    
            // for each possible edge
            for(int i=r; i>=(-r); i--){
                for(int j=(-r); j<=r; j++){
                    // i&j are relative coordinates, ii&jj are absolute coordinates
                    int ii = i+iC, jj = j+jC;

                    // The cell itself (when i=0 and j=0) is here considered but never added to the queue due to the logic of the BFS
                    if( Planning::map->checkPointInActualMap(Point2<int>( jj*Planning::map->getCellSize(), ii*Planning::map->getCellSize() ))
                        ||
                        (  Planning::map->checkCellInMap(ii, jj) && Planning::map->getCellType(ii, jj) == GATE ) ){

                        if( Planning::map->getCellType(ii, jj) != OBST 
                            && (dist<initialDistAllowed || map->getCellType(ii, jj) != BODA )
                            ){ 
                            if(ii==endC.y() && jj==endC.x()){
                                found++;
                            }
                            double myDist = computedDistances[(i+r)*side + (j+r)];
                            // if not visited or previous bigger distance
                            if( equal(distances[ii][jj], baseDistance, 0.001) || distances[ii][jj] > dist + myDist ){
                                distances[ii][jj] = dist + myDist;
                                parents[ii][jj] = cell.point();

                                toProcess.push(Configuration2<int>(jj, ii, 0.0));
                            }
                        }
                    }
                }
            }
        }

        cout << "End iteration" << endl;
        // reconstruct the vector of parents of the cells in the minPath
        vector<Point2<int> > computedParents;

        if(found==0){
            cout << "\n\n\t\tDestination of minPath not reached ! ! !\nSegment from: " << startP << " to " << endP << "\nNo solution exist ! ! !\n\n";
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "MinPath can't reach the destination.", __LINE__, __FILE__);
        } else {
            // computing the vector of parents
            computedParents.push_back(endP);
            Point2<int> p = endC;
            while( p!=startC ){
                p = parents[p.y()][p.x()];

                // conversion from cell of the grid to point of the system (map)
                computedParents.push_back( Point2<int>( (p.x()+0.5)*map->getPixX(), (p.y()+0.5)*map->getPixY() ));
            }
            reverse(computedParents.begin(), computedParents.end()); // I apply the inverse to have the vector from the begin to the end.
        }
        return(computedParents);
    }

    /*! \brief Compute the sector of an angle.

        \param[in] d The initial angle in radiants.
        \rreturn The sector of the angle
    */
    int angleSector(const double & d){
        return( (int)(d/(2*M_PI/nAngles)) );
    }

    vector<Point2<int> > minPathTwoPointsInternalAngles(
                            const Point2<int> & startP, const Point2<int> & endP, 
                            double *** distances, int **** parents,
                            const double initialDir)
    {
        // reset base values
        resetDistanceMap(distances);

        // generate and check the base elements
        // P=point, C=cell
        Point2<int> startC(startP.x()/map->getPixX(), startP.y()/map->getPixY());
        Point2<int> endC(endP.x()/map->getPixX(), endP.y()/map->getPixY());
        if(map->getCellType(startC.y(), startC.x()) == OBST){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "The start position of a segment of the minPath is inside an Obstacle!", __LINE__, __FILE__);
        }
        if(map->getCellType(endC.y(), endC.x()) == OBST){
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "The end position of a segment of the minPath is inside an Obstacle!", __LINE__, __FILE__);
        }

        // initialization of BFS (adding the first configurations: one or nAngles)
        queue<Configuration2<int> > toProcess;
        if(equal( initialDir, baseDir, 0.001)){ // never used and probably this option has no reason to exist.
            // initialize a direction with dist=0 for each angle sector
            for(int q=0; q<nAngles; q++){
                toProcess.push(Configuration2<int>(startC, (q+0.5)*(2*M_PI/nAngles) ));
                distances[startC.y()/*i=y()*/][startC.x()/*j=x()*/][q] = 0.0;
                parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][q][0] = startC.y();
                parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][q][1] = startC.x();
                parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][q][2] = q;
            }
        } else{
            toProcess.push(Configuration2<int>(startC, initialDir));
            distances[startC.y()/*i=y()*/][startC.x()/*j=x()*/][ angleSector(initialDir) ] = 0.0;
            parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][ angleSector(initialDir) ][0] = startC.y();
            parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][ angleSector(initialDir) ][1] = startC.x();
            parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/][ angleSector(initialDir) ][2] = angleSector(initialDir);
        }

        int found = 0;

        // precompute the computation of the distances and the angle in the square of edges around the cell of interest
        const int r = range+10; //range from class variable (default=3)
        const int side = 2*r+1;
        double computedDistances[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
        double computedAngles[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
        
        cout << "\ncomputedAngles\n\t\b\b";
        for(int j=(-r); j<=r; j++){
            cout << j << "\t\b\b";
        }
        for(int i=r; i>=(-r); i--){
            cout << "\n" << i << ":\t\b\b";
            for(int j=(-r); j<=r; j++){

                computedDistances[(i+r)*side + (j+r)]  = sqrt( pow(i,2) + pow(j,2) );
                computedAngles[(i+r)*side + (j+r)] = (Point2<int>(0, 0).th( Point2<int>(j, i) )).toRad();
                
                cout << (int)(computedAngles[(i+r)*side + (j+r)]*180.0/M_PI) << "\t\b\b";
            }
        }

        // start iteration of the BFS
        while( !toProcess.empty() && found<=foundLimitAngles ){
            // for each cell from the queue
            Configuration2<int> cell = toProcess.front();
            toProcess.pop();

            int iC = (int)cell.y(), jC = (int)cell.x(); // i and j of the cell
            double thC = cell.angle().toRad();            // th of the cell
            int sectC = angleSector(thC);               // sector of the angle of the cell
            double dist = distances[iC][jC][sectC];     // distance up to that cell
                    
            // for each possible edge around the actual cell
            for(int i=r; i>=(-r); i--){
                for(int j=(-r); j<=r; j++){
                    // i&j are relative coordinates, ii&jj are absolute coordinates in the grid
                    int ii = i+iC, jj = j+jC;
                    double myDist =  computedDistances[(i+r)*side + (j+r)];
                    double myAngle = computedAngles[   (i+r)*side + (j+r)];
                    int mySect = angleSector(myAngle);

                    // In case of first segment, it is also neccessary to check that the angle of the new point is more or less correct respect to the previous one.
                    if(fabs( myAngle-thC ) < angleRange ){
                        // The cell itself (when i=0 and j=0) is here considered but never added to the queue due to the logic of the BFS
                        if( Planning::map->checkPointInActualMap(Point2<int>( jj*Planning::map->getCellSize(), ii*Planning::map->getCellSize() ))
                            ||
                            (  Planning::map->checkCellInMap(ii, jj) && Planning::map->getCellType(ii, jj) == GATE ) ){
                            // cell is allowed
                            if( map->getCellType(ii, jj) != OBST
                                // && (dist<initialDistAllowed || map->getCellType(ii, jj) != BODA )
                                ){ 

                                if(ii==endC.y() && jj==endC.x()){
                                    found++;
                                }

                                // if not visited or previous bigger distance
                                if(equal(distances[ii][jj][mySect], baseDistance, 0.001) ||  distances[ii][jj][mySect] > dist + myDist ){
                                    // Update its value and save for future processing
                                    distances[ii][jj][mySect]  = dist + myDist;
                                    parents[ii][jj][mySect][0] = iC;
                                    parents[ii][jj][mySect][1] = jC;
                                    parents[ii][jj][mySect][2] = sectC;

                                    toProcess.push(Configuration2<int>(jj, ii, myAngle));
                                }
                            }
                        }
                    }
                }
            }
        }

        // reconstruct the vector of parents of the cells in the minPath
        vector<Point2<int> > computedParents;

        if(found==0){
            cout << "\n\n\t\tDestination of minPath with angles not reached ! ! !\nSegment from: " << startP << " to " << endP << " has no solution ! ! !\n\n";
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "MinPath with angles can't reach the destination.", __LINE__, __FILE__);
        } else {
            // choose the best direction path up to the end point
            double bestDist = DInf;
            int bestDir = -1;
            // cout << "The final distance for the point: " << endP <<  " from the point " << startP << endl << "AKA the cells: " << endC << " and " << startC << endl;

            for(int q=0; q<nAngles; q++){
                double finalDist = distances[endC.y()][endC.x()][q];
                // cout << q << ": Dir [" << q*45 << ", " << q*45+44 << "]: " << finalDist << endl;
                if( !equal(finalDist, baseDistance, 0.0001) && bestDist>finalDist ){
                    bestDist = finalDist;
                    bestDir = q;
                }
            }
            if(bestDir==-1){
                throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Something strange went wrong in the computation of the vector of parents in the minPath.", __LINE__, __FILE__);
            } else{
                // cout << "Chosen the best! is " << bestDir << " with: " << bestDist << endl;
                
                computedParents.resize(0);
                // computing the vector of parents
                computedParents.push_back(endP);
                int * v = new int[3] { endC.y(), endC.x(), bestDir };
                while(!( v[0]==startC.y() && v[1]==startC.x() )){
                    v = parents[v[0]][v[1]][v[2]];

                    // conversion from cell of the grid to point of the system (map)
                    computedParents.push_back( Point2<int>( (v[1]+0.5)*map->getPixX(), (v[0]+0.5)*map->getPixY() ));
                }
                reverse(computedParents.begin(), computedParents.end()); // I apply the inverse to have the vector from the begin to the end.
                // delete[] v; doesn't need to be deleted now, will be done soon.
            }
        }

        return(computedParents);
    }


    /*! \brief Converts an integer into the vector of its digits. The result is inverse respect to the given integer.

        \param[in] c The to split into the vector.
        \param[out] v The vector where the split will be saved.
    */
    void intToVect(int c, vector<int> & v){
        v.resize(0);
        while(c>0){
            v.push_back(c%10);
            c /= 10;
        }
        reverse(v.begin(), v.end());
    }

    /*! \brief It reset, to the given value, the matrix of distances given, to compute again the minPath search.

        \param[out] distances It is the array that need to be initialized.
        \param[in] value The value to be set.
    */
    void resetDistanceMap(double ** distances, const double value){
        for(int i=0; i<map->getDimY(); i++){
            for(int j=0; j<map->getDimX(); j++){
                distances[i][j] = value;
            }
        }
    }

    /*! \brief It reset, to the given value, the matrix of distances given, to compute again the minPath search.

        \param[out] distances It is the array that need to be initialized.
        \param[in] value The value to be set.
    */
    void resetDistanceMap(double *** distances, const double value){
        for(int i=0; i<map->getDimY(); i++){
            for(int j=0; j<map->getDimX(); j++){
                for(int k=0; k<nAngles; k++){
                    distances[i][j][k] = value;
                }
            }
        }
    }

    /*! \brief It extracts from the given vector of vector of points, a subset of points that always contains the first one and the last one of each vector.

        \param[in] n The n number of points to sample.
        \param[in] points The vector of vector of points to be selected.
        \returns The vector containing the subset of n points.
    */
    vector<Point2<int> > sampleNPoints(const vector<vector<Point2<int> > > & vvp, const int n){
        cout << "sample0\n" << flush;
        vector<Point2<int> > vp;
        if(n < (int)vvp.size()+1){
            cout << "\n\nSampling N points: N is too small (at least vvp.size()+1 is required). . .\n\n";
        } else{
        cout << "sample1\n" << flush;
            int totalSize = 0;
            for(auto el : vvp){
                totalSize += el.size()-1;
            }
            float step = (totalSize-1)*1.0/(n-2);

        cout << "sample2\n" << flush;
            int tmpSize = 0;
            for(float i=0, v=0; (int)i<totalSize; i+=step){
                if((unsigned int)i < vvp[v].size()+tmpSize){
                    vp.push_back(vvp[v][(int)i-tmpSize]);        
                } else{
                    tmpSize += vvp[v].size();
                    v++;
                    if(v>=vvp.size()){
                        break;
                    }
                    vp.push_back( vvp[v][0] );
                }
            }
        cout << "sample3\n" << flush;
            vp.push_back( vvp.back().back() );
        }
        cout << "sample4\n" << flush;
        return(vp);
    }

    /*! \brief It extracts from the given vector of points, a subset of points that always contains the first one and the last one.

        \param[in] n The number of points to select exept the extremes, it must be greater or equal than 2.
        \param[in] points The vector of points to be selected.
        \returns The vector containing the subset of n points.
    */
    vector<Point2<int> > sampleNPoints(const vector<Point2<int> > & points, const int n){
        vector<Point2<int> > vp;
        if(n >= (int)points.size() || points.size()==2){
            vp = points;
        } else if (points.size() > 2){
            float step = (points.size()-1)*1.0/n;
            for(int i=0; i<n-1; i++){
                vp.push_back(points[ (int)i*step ]);
            }
            vp.push_back(points.back());
        } else{
            cout << "Invalid value of n and dimension of the vector.\n\n";
        }
        return(vp);    
    }

    /*! \brief It extracts from the given vector of points, a subset of points that always contains the first one and the last one.

        \param[in] step The distance (counted as cells) from the previous to the next cell, it must but >=2 to have a reason.
        \param[in] points The vector of points to be selected.
        \returns The vector containing the subset of points, each step cells.
    */
    vector<Point2<int> > samplePointsEachNCells(const vector<Point2<int> > & points, const int step){
        vector<Point2<int> > vp;
        if(step<=1 || points.size()==2){
            vp = points;
        } else if (points.size() > 2){
            for(unsigned int i=0; i<points.size()-1; i+=step){
                vp.push_back(points[ i ]);
            }
            vp.push_back(points.back());
        } else{
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Invalid value of step and dimension of the vector.", __LINE__, __FILE__);
        }
        return(vp);    
    }

    #define SCALE 1000.0
    /*! \brief Convert a vector of point to a path, from Enrico's notation to Paolo's notation.

        \param[in] vp The sorce vector.
        \param[out] path The destination path.
    */
    void fromVcToPath(vector<Configuration2<double> > & vc, Path & path){
        if(vc.size()>=2){
            path.resize(0);
            uint i=0;
            Pose pose;
            double k=0.1, th, th1, s=0.0; // k will be computed thanks to the dubins

            vc[0].invert();
            th = vc[0].angle();
            path.add(Pose( 0, vc[0].x()/SCALE, vc[0].y()/SCALE, th, 0));

            bool normalScenario = true;
            for(i=1; i<vc.size(); i++){
                vc[i].invert();

                th1 = th;
                th = vc[i].angle();
                
                if(fabs(th-th1) > M_PI){
                    normalScenario = false;
                }

                if(equal(th, th1, 0.001)){
                    k = 0.0;
                } else if(th > th1){    //right k=-1
                    k = SCALE*ROB_KMAX*(normalScenario ?  1 : (-1));
                } else{                 //left k=+1
                    k = SCALE*ROB_KMAX*(normalScenario ?  (-1) : 1);
                }

                if(!normalScenario){
                    normalScenario = true;
                }
                
                s += vc[i-1].point().distance(vc[i].point())/SCALE;

                pose = Pose( s, vc[i].x()/SCALE, vc[i].y()/SCALE, th, k);
                path.add(pose);
            }
            path.points[path.points.size()-1].kappa = 0.0;
            path.points[path.points.size()-2].kappa = 0.0;
            path.points[path.points.size()-3].kappa = 0.0;
            path.points.pop_back();

            cout << "Path elements:\n";
            for(i=0; i<path.size(); i++){
                cout << "pose " << i+1 << "°: " << path.points[i].string().str() << endl;
            }
        } else{
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Impossible to convert vector to path, dimension less than 2.", __LINE__, __FILE__);
        }
    }


    template<class T>
    bool check_dubins_D (Dubins<T>& D){
        Tuple<Tuple<Configuration2<T> > > vPDub=D.splitIt(10);
        bool ok=true;
        for (int j=0; j<3 && ok; j++){ 
            for (int k=0; (k<(int)(vPDub[j].size())-1) && ok; k++){    
                if(!(Planning::map->checkPointInActualMap(vPDub[j][k].point()) || Planning::map->getPointType(vPDub[j][k].point())==GATE)) {
                    cout << "Point of index " << k << " out of map: " << vPDub[j][k] << endl;
                    ok=false;
                } 
                else if (Planning::map->checkSegment(vPDub[j][k].point(), vPDub[j][k+1].point())){
                    cout << "Segment through obstacles: " << vPDub[j][k].point() << " " << vPDub[j][k+1].point() << endl;
                    ok=false;
                }
            }
            if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1].point())==OBJ_TYPE::OBST){
                cout << "Last point is on obstacle: " << vPDub[j].back().point() << endl;
                ok=false;
            }
            else if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1].point())==OBJ_TYPE::OUT_OF_MAP){
                cout << "Last point is out of map: " << vPDub[j].back().point() << endl;
                ok=false;
            }
        }
        return ok;
    }

    template<class T>
    bool check_dubins_DS (DubinsSet<T>& DS){
        bool ok=true;
        for (int i=0; i<DS.getSize() && ok; i++){
            Dubins<T> app=DS.getDubins(i);
            ok=check_dubins_D(app);
        }
        return ok;
    }

    #define INCREASE 20 //mm
    #define SCRAP 3 //Number of points to be scrapped at the extremities of each segment of path
    extern const double DEGTORAD;

    bool compute_roundabout_dubins( DubinsSet<double>& new_DS,
                                    Configuration2<double> _start, 
                                    const vector<Configuration2<double> >& vC,
                                    uint& vC_id, //I want to return the last pos I checked
                                    bool gate=false){
        bool ok=false;
        int scrap = (gate ? 0 : SCRAP);
        for (vC_id=scrap; (vC_id<(vC.size()-scrap) && !ok); vC_id++){
            Configuration2<double> end=vC[vC_id]; 
            Configuration2<double> start=_start;

            Angle offset(-3.0*M_PI/4.0, Angle::RAD);
            start.offset_angle(offset);
            //Then I try to find a DubinsSet from the victim to a point in V2 with different orientation in \phi-60, \phi+60
            
            for (int i=0; i<(offset.toRad()*2.0/(10.0*::DEGTORAD)+1) && !ok; i++){
                start.offset_angle(Angle(20.0*::DEGTORAD, Angle::RAD));
                Point2<double> intermediate=::circline(INCREASE, start, 0);
                do {
                    new_DS=DubinsSet<double> (_start, end, Tuple<Point2<double> > (vector<Point2<double> >{intermediate}), ROB_KMAX);
                    ok=check_dubins_DS(new_DS);
                    // intermediate.offset(INCREASE, start.angle());
                } while(Planning::map->checkPointInActualMap(intermediate.offset(INCREASE, start.angle())) && !ok);

            }
        }
        return ok;
    }

    //TODO CHECK vP2_pos
    template<class T> 
    DubinsSet<double> victims_dubins (const vector<Configuration2<T> >& vC1,
                                      const vector<Configuration2<T> >& vC2,
                                     uint& vC1_pos,
                                     uint& vC2_pos)
    {
        uint vC1_id=0;
        uint vC2_id=0; 

        DubinsSet<double> DS;
        bool ok=false;

        for (vC1_id=SCRAP; vC1_id<vC1.size()-SCRAP && !ok; vC1_id++){
            for (vC2_id=vC2.size()-SCRAP; vC2_id>SCRAP && !ok; vC2_id--){
                DS=DubinsSet<double> (vC1[vC1_id], vC2[vC2_id], Tuple<Point2<double> >(vector<Point2<double> >{vC1.back().point()}), ROB_KMAX);
                ok=check_dubins_DS(DS);
            }
        }
        if (ok){
            cout << DS << endl;
            vC1_pos=vC1_id;
            vC2_pos=vC2_id;
        }
        if (!ok){//Oh boy we are in truble. 
            //First I try to create a Dubins from a point to the victim.
            cout << "TRYING MORE DUBINS" << endl;
            DS.clean();
            Dubins<double> D;
            Configuration2<double> end=vC1.back(); //I want to arrive to the victim with an angle such as the direction
            // for (vC1_id=1; vC1_id<vC1.size()-1 && !ok; vC1_id++){
            for (vC1_id=SCRAP; vC1_id<vC1.size()-SCRAP && !ok; vC1_id++){
                D=Dubins<double> (vC1[vC1_id], end, ROB_KMAX);
                ok=check_dubins_D(D);
            }
            if (ok){ 
                DubinsSet<double> new_DS;
                ok=compute_roundabout_dubins(new_DS, D.end(), vC2, vC2_id);
                if (ok){ //Add everything at the end.
                    vC1_pos=vC1_id;
                    vC2_pos=vC2_id;
                    DS.addDubins(&D);
                    DS.join(&new_DS);
                }
            }
        }
        if (!ok){
            throw MyException<string> (GENERAL, ("No DubinsSet could be computed for victim at: "+vC1.back().to_string().str()), __LINE__, __FILE__);
        }
        return DS;
    }


    template<class T>
    DubinsSet<double> start_end_dubins(const Configuration2<double>& anchorPoint, 
                                    const vector<Configuration2<T> >& vConfs,
                                    uint& pos,
                                    const bool start)
    {
        DubinsSet<double> ds;
        Dubins<double> dub;
        bool ok=false;
        const uint size=vConfs.size()-1;
        for (uint i=size-2; (i>0 && !ok); i--){ //I continue until the points are empty or until a feasible dubins is not found. 
            ok=true; //Need to reset this each loop
            if (start){
                dub=Dubins<double>(anchorPoint, vConfs[i], ROB_KMAX);
            }
            else {
                dub=Dubins<double>(vConfs[size-i-1], anchorPoint, ROB_KMAX);
            }

            if(dub.getId()<0){ //If Dubins could not be computed with given points set ok to false.
                ok = false;
            }
            else { //Else if a Dubins could be computed then check if it's valid.  
                ok=check_dubins_D(dub);
                if (ok){ //If it is, save the Dubins inside the set, and save the position. Then we are done
                    ds.addDubins(&dub);
                    if (start){
                        pos=i;
                    }
                    else {
                        pos=size-i;
                    }
                }
            }
        }
        if (!ok) { //If even after all points in the vector no Dubins could be found, then try some more points. 
            if(start){//If we are working on the start, then the first point is from where I want to start from and get back to.
                ok=compute_roundabout_dubins(ds, anchorPoint, vConfs, pos);
            }//I should never need to find a way around for the end and, if I do, it's probably going to be too expensive 
            else {
                cout << "TRYING MORE DUBINS" << endl;
                for (int i=0; i<(int)(vConfs.size()-SCRAP) && !ok; i++){
                    ok=compute_roundabout_dubins(ds, vConfs[i], vector<Configuration2<double> >{anchorPoint}, pos, true);
                }
            }
            if (!ok){
                ds=DubinsSet<double> ();
            }
        }
        cout << "End dubins start/end" << endl;
        if (ok){
            return ds;
        }
        else {
            pos=0;
            throw MyException<string> (EXCEPTION_TYPE::GENERAL, "No feasible path was found.", __LINE__, __FILE__);
        }
    }

    vector<Configuration2<double> > vvCtovC(Tuple<Tuple<Configuration2<double> > > vv)
    {
        vector<Configuration2<double> >v={};
        for (auto a : vv){
            for (auto b : a){
                v.push_back(b);
            }
        }
        return v;
    }

    vector<Configuration2<double> > vvvCtovC(Tuple<Tuple<Tuple<Configuration2<double> > > > vvv)
    {
        vector<Configuration2<double> > v={};
        for (auto a : vvv){
            for (auto b : a){
                for (auto c : b){
                    v.push_back(c);
                }
            }
        }
        return v;
    }

    Angle compute_final_angle (Configuration2<double> gate){
        double dx=Planning::map->getLengthX()-gate.x();
        double dy=Planning::map->getLengthY()-gate.y();
        if (dx<gate.x()){
            if (dx<gate.y()){
                if (dx<dy){
                    return Angle(0.0, Angle::RAD);
                }
                else {
                    return Angle(M_PI/2.0, Angle::RAD);
                }
            }
            else {
                if (gate.y()<dy){
                    return Angle(3.0*M_PI/2.0, Angle::RAD);
                }
                else {
                    return Angle(M_PI/2.0, Angle::RAD);
                }
            }
        }
        else {
           if (gate.x()<gate.y()){
                if (gate.x()<dy){
                    return Angle(M_PI, Angle::RAD);
                }
                else {
                    return Angle(M_PI/2.0, Angle::RAD);
                }
            }
            else {
                if (gate.y()<dy){
                    return Angle(3.0*M_PI/2.0, Angle::RAD);
                }
                else {
                    return Angle(M_PI/2.0, Angle::RAD);
                }
            } 
        }
    }

    void inter_victims(vector<vector<Configuration2<double> > >& vvConfs,
                            vector<int>& vI,
                            DubinsSet<double>& path,
                            vector<DubinsSet<double> >& victimV)
    {
        Configuration2<double> right, left;
        cout << "vvConfs.size(): " << vvConfs.size() << " vI.size: " <<  vI.size() << endl; 
        for (int i=0; i<(int)vvConfs.size() && i<(int)(vI.size()/2) && victimV.size()>0; i++){ 
            cout << "Adding victim: " << i << endl;
            try{
                path.join(&victimV[i]);
            }
            catch(exception& e){
                cerr << "inter_victims: " << e.what() << endl;
            }    
            if (vI[i*2]<vI[i*2+1]){ //Right < Left
                right= vvConfs[i][vI[i*2]];
                left= vvConfs[i][vI[i*2+1]];
                cout << "Trying Dubins from: " << right << " to " << left << endl;
                Dubins<double> D = Dubins<double> (left, right, ROB_KMAX);
                if (!equal(D.length(), DInf) && check_dubins_D(D)){
                    try{
                        path.addDubins(&D);
                    }
                    catch (exception& e){
                        cerr << e.what() << endl;
                    }
                }
                else {
                    cerr << "No Dubins could be created between Configurations " << right << " and " << left << endl;
                }
                
            }

        } 

    } 

    /*!
     * Function to compute `Dubins` between the various points in the path.
     * \param[in] start `Configuration2` that is the starting configuration. 
     * \param[in/out] vvConfs Vector of vectors of `Configuration2` that are the points of the path.
     */
    void plan_dubins( const Configuration2<double>& _start,
                        vector<vector<Configuration2<double> > >& vvConfs)
    {
        cout << "conf in plan: " << _start << endl ;
        DubinsSet<double> path;
        //Create Dubins for first path.
        DubinsSet<double> start;
        vector<uint> lenghts={}; for (auto a : vvConfs) {lenghts.push_back(a.size());} //TODO do I use this??
        vector<int> vI;

        uint start_pos=0, end_pos=0;
        try{
            start=start_end_dubins(_start, vvConfs[0], start_pos, true);
        } catch(exception& e){
            cout << e.what() << endl << flush;
            start=DubinsSet<double>();
        }
        if (start.getLength()!=DInf){
            cout << "Starting Dubins found." << endl << flush;
            uint app=start_pos;
            vector<Configuration2<double> > new_points=vvvCtovC(start.splitIt((ROB_PIECE_LENGTH>15.0) ? 15.0 : ROB_PIECE_LENGTH));
            start_pos=new_points.size()-1;
            for (uint i=app+1; i<vvConfs[0].size(); i++){
                new_points.push_back(vvConfs[0][i]);
            }
            vvConfs[0]=new_points;

            vI.push_back(start_pos);
        } 

        draw(vvConfs, "Map Start");

        //Create Dubins for victims
        vector<DubinsSet<double> >victimV; //TODO do i use this??
        DubinsSet<double> victim;

        for (int i=0; i<(int)(vvConfs.size())-1; i++){
            cout << "Trying Dubins for victim: " << i << endl;
            try{
                victim=victims_dubins(vvConfs[i], vvConfs[i+1], start_pos, end_pos);
            } 
            catch(exception& e){
                cout << e.what() << endl << flush;
                victim=DubinsSet<double>();
            }
            if (victim.getLength()!=DInf){
                victimV.push_back(victim);
                cout << "Dubins for victim: " << i+1 << " Found." << endl << flush;
                //First vector
                vector<Configuration2<double> > V1={};
                vector<Configuration2<double> > new_points=vvCtovC(victim.getDubins(0).splitIt(ROB_PIECE_LENGTH));
                for (uint j=0; j<start_pos; j++){
                    V1.push_back(vvConfs[i][j]);
                }
                for (uint j=0; j<new_points.size(); j++){
                    V1.push_back(new_points[j]);
                }
                //Second vector
                vector<Configuration2<double> > V2={};
                new_points=vvvCtovC(DubinsSet<double>(victim.getDubinsFrom(1)).splitIt(ROB_PIECE_LENGTH));
                for (uint j=0; j<new_points.size(); j++){
                    V2.push_back(new_points[j]);
                }
                for (uint j=end_pos+1; j<vvConfs[i+1].size(); j++){
                    V2.push_back(vvConfs[i+1][j]);
                } 
                vvConfs[i]=V1;
                vvConfs[i+1]=V2;
                vI.push_back(start_pos);
                vI.push_back(end_pos);

                draw(vvConfs, ("map victim "+to_string(i+1)));
            }
        }

        //Create Dubins for final path.
        DubinsSet<double> end;
        cout << "Creating Dubins for end" << endl;
        try{
            end=start_end_dubins(Configuration2<double>(vvConfs.back().back().point(), compute_final_angle(vvConfs.back().back())),
                                 vvConfs.back(), end_pos, false); //TODO find better implementation for final angle
        }
        catch (exception& e){
            cout << e.what() << endl << flush;
            end=DubinsSet<double>();
        }
        if (end.getLength()!=DInf){
            cout << "Ending Dubins found." << endl << flush;

            vector<Configuration2<double> > new_points=vvvCtovC(end.splitIt(ROB_PIECE_LENGTH));
            cout << "new_points: " << new_points.size() << " pos: " << end_pos << " vvConfs[-1]: " << vvConfs.back().size() << endl << flush;

            vector<Configuration2<double> > app;
            for (uint i=0; i<end_pos; i++){
                app.push_back(vvConfs.back()[i]);
            }
            for (uint i=0; i<new_points.size(); i++){
                app.push_back(new_points[i]);
            }
            vvConfs[vvConfs.size()-1]=app;
            cout << "Size: " << (vvConfs.back()).size() << endl;
            vI.push_back(end_pos);    
            draw(vvConfs, "map end");
        }

        cout << "BUILDING DUBINS THROUGH VICTIMS" << endl;
        try {
            path.join(&start);
            cout << "Start: " << start << endl << endl << endl;
            cout << path << endl;
            inter_victims(vvConfs, vI, path, victimV);
            path.join(&end);
        }
        catch (exception& e){
            cerr << e.what() << endl;
        }
        cout << "FINISHED BUILDING DUBINS THROUGH VICTIMS" << endl;


        cout << "End of using Dubins" << endl << flush;
        cout << "vI: " << vI.size() << endl;

        //Draw points
        vector<Configuration2<double> > left, right;
        cout << "vvConfs.size: " << vvConfs.size() << endl;
        cout << "(int)(vI.size()/2): " << (int)(vI.size()/2) << endl;
        for (uint i=0; i<(uint)(vI.size()/2) && i<(uint)(vvConfs.size()); i++){ 
            right.push_back(vvConfs[i][vI[i*2]]);
            left.push_back(vvConfs[i][vI[i*2+1]]);
            cout << i << " Right: " << right.back() << " left: " << left.back() << endl;
        }
        cout << "About to draw" << endl;
        draw(vvConfs, left, right, "SD");
        cout << "End drawing " << path.getSize() << endl;
    }
}
