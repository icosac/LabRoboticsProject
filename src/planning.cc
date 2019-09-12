#include"planning.hh"

namespace Planning {
    Mapp* map;
    Configuration2<double> conf;
    const double angleRange = 15*M_PI/180;
    const int nAngles = 60;
    const int range = 5;    


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

        // localize the robot
        pair<Configuration2<double>, Configuration2<double> > p = ::localize(img, true);
        conf = p.first;
        vp.push_back( Point2<int>( (int)conf.x(), (int)conf.y()) ); //robot initial location.

        // update and show the created map
        cout << "plan3\n" << flush;
        Planning::map->getVictimCenters(vp);
        cout << "plan3 bis\n" << flush;
        Planning::map->getGateCenter(vp);


        Mat imageMap = Planning::map->createMapRepresentation();
        cout << "Created map" << endl << flush;

        imwrite("data/computePlanning.jpg", imageMap);
        #if defined WAIT || defined SHOW_MAP
            namedWindow("Map 0", WINDOW_NORMAL);
            imshow("Map 0", imageMap);
            mywaitkey();
        #endif
        cout << "plan4\n" << flush;

        // search for the shortest path
        // #define BEST
        #ifdef BEST 
            vector<vector<Point2<int> > > vvp = minPathNPointsWithChoice(vp, 5);
        #else
            vector<vector<Point2<int> > > vvp = minPathNPoints(vp, true);
        #endif
        cout << "plan5\n" << flush;
        // show the path and prepare it for the return
        cout << "aaa\n";
        vector<Point2<int> > cellsOfPath = new_function(vvp);
        // vector<Point2<int> > cellsOfPath = sampleNPoints(vvp);
        cout << "aaa\n";
        Planning::map->imageAddPoints(imageMap, cellsOfPath);
        cout << "aaa\n";
        Planning::map->imageAddSegments(imageMap, cellsOfPath);
        cout << "Created map 2" << endl << flush;

        imwrite("data/computePlanning.jpg", imageMap);
        #if defined WAIT || defined SHOW_MAP
            namedWindow("Map 1", WINDOW_NORMAL);
            imshow("Map 1", imageMap);
            mywaitkey();
        #endif
        

        // Add Dubins to improve the path
        cout << "Trying to use dubins" << endl << flush;
        // Planning::plan_best(conf, vvp);
        // cout << "Found best dubins" << endl;

        // // vector<Point2<int> > cellsOfPath = new_function(vvp);
        // cellsOfPath = new_function(vvp);
        // // vector<Point2<int> > cellsOfPath = sampleNPoints(vvp);
        // // cellsOfPath = sampleNPoints(vvp);

        // cout << "plan6\n" << flush;
        // cout << "\tCellsOfPath size: " << cellsOfPath.size() <<endl;
        // cout << "plan7\n" << flush;

        // // imageMap = Planning::map->createMapRepresentation();
        // // cout << "Created map" << endl << flush;
        // Planning::map->imageAddPoints(imageMap, cellsOfPath);
        // cout << "Created map 1" << endl << flush;
        // Planning::map->imageAddSegments(imageMap, cellsOfPath);
        // cout << "Created map 2" << endl << flush;

        // imwrite("data/computePlanning.jpg", imageMap);
        // #if defined WAIT || defined SHOW_MAP
        //     namedWindow("Map 2", WINDOW_NORMAL);
        //     imshow("Map 2", imageMap);
        //     mywaitkey();
        // #endif

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
        cout << "CONVEXHULLFILE: " << (sett->baseFolder+sett->convexHullFile) << endl;
        cout << "create2 a\n" << flush;
        
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
                cerr << "Warning: Loaded no obstacles for the creating of the map. " << __LINE__ << " " << __FILE__ << endl;
            }
            cout << "create3\n" << flush;

        cout << "create4\n" << flush;
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

    int getNPoints() { return nPoints; }


    int ** allocateAAInt(const int a, const int b){
        int ** arr = new int*[a];
        for(int i=0; i<a; i++){
            arr[i] = new int[b];
        }
        return(arr);
    }

    int *** allocateAAAInt(const int a, const int b, const int c){
        int *** arr = new int**[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAInt(b, c);
        }
        return(arr);
    }

    int **** allocateAAAAInt(const int a, const int b, const int c, const int d){
        int **** arr = new int***[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAAInt(b, c, d);
        }
        return(arr);
    }

    double ** allocateAADouble(const int a, const int b){
        double ** arr = new double*[a];
        for(int i=0; i<a; i++){
            arr[i] = new double[b];
        }
        return(arr);
    }

    double *** allocateAAADouble(const int a, const int b, const int c){
        double *** arr = new double**[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAADouble(b, c);
        }
        return(arr);
    }

    double **** allocateAAAADouble(const int a, const int b, const int c, const int d){
        double **** arr = new double***[a];
        for(int i=0; i<a; i++){
            arr[i] = allocateAAADouble(b, c, d);
        }
        return(arr);
    }

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
        \returns A vector of vector of points along the path (one for each cell of the grid of the map). Each vector is the best path for one connection, given n points there are n-1 connecctions.
    */
    vector<vector<Point2<int> > > minPathNPointsWithChoice(const vector<Point2<int> > & vp, const double bonus){

        //allocate
        double ** distances = new double*[map->getDimY()];
        Point2<int> ** parents = new Point2<int>*[map->getDimY()];
        for(int i=0; i<map->getDimY(); i++){
            // the initializtion is to -1
            distances[i] = new double[map->getDimX()];
            parents[i] = new Point2<int>[map->getDimX()];
        }

        //compute the simple minPath between all pairs of targets
        int n = vp.size();
        vector< vector< vector<Point2<int> > > > vvvp(n, vector< vector<Point2<int> > >(n));
        vector< vector<int> > vvDist(n, vector<int>(n));

        cout << "\nTable of distances:\n";
        for(int i=0; i<n; i++){
            cout << i << "\t";
        }

        for(int i=0; i<n-1; i++){
            cout << i << ":\t" << flush;
            for(int j=0; j<i; j++) cout << "\t";
            for(int j=i+1; j<n; j++){
                //compute the minPath
                vvvp[i][j] = minPathTwoPointsInternal(vp[i], vp[j], distances, parents);

                //compute the overall distance along the computed path and store it
                double dist = 0.0;
                for(unsigned int q=0; q<vvvp[i][j].size()-1; q++){
                    dist += vvvp[i][j][q].distance( vvvp[i][j][q+1] );
                }
                cout << dist << "\t" << flush;
                vvDist[i][j] = dist;
                vvDist[j][i] = dist;
            }
        }

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
            cout << el << ",\tThe cost is: ";
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
        cout << "\nThe best cost is: " << bestCost << ", generated by the disp: " << bestDisp << endl;

        //prepare the return values
        vector< vector<Point2<int> > > vvp;
        intToVect(bestDisp, targets);
        if(targets.size()==0){
            vvp.push_back( vvvp[0][ n-1 ]);
        } else{
            vvp.push_back( vvvp[0][targets.front()] );
            for(unsigned int q=0; q<targets.size()-1; q++){
                vvp.push_back(vvvp[ targets[q] ][ targets[q+1] ]);
            }
            vvp.push_back( vvvp[ targets.back() ][ n-1 ] );
        }

        //delete
        for(int i=0; i<map->getDimY(); i++){
            delete [] distances[i];
            delete [] parents[i];
        }
        delete[] distances;
        delete[] parents;

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
        cout << "min0\n" << flush;

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
            // deleteAAA(distances, map->getDimY(), map->getDimX());
            // deleteAAAA(parents, map->getDimY(), map->getDimX(), nAngles);
            i++;
        }

        // Normal version
        //allocate
        double ** distances = allocateAADouble(map->getDimY(), map->getDimX());
        Point2<int> ** parents = allocateAAPointInt(map->getDimY(), map->getDimX());

        // function
        for(; i<(int)(vp.size())-1; i++){
            vvp.push_back( minPathTwoPointsInternal(vp[i], vp[i+1], distances, parents));
        }

        //delete
        // deleteAA(distances, map->getDimY());
        // deleteAA(parents, map->getDimY());
        
        cout << "min5\n" << flush;

        return(vvp);
    }

    /*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
        \details The function is based on a Breadth-first search (BFS).

        \param[in] p0 The source point.
        \param[in] p1 The destination point.
        \returns A vector of points along the path (one for each cell of the grid of the map).
    */
    vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1){
        //allocate
        double ** distances = new double*[map->getDimY()];
        Point2<int> ** parents = new Point2<int>*[map->getDimY()];
        for(int i=0; i<map->getDimY(); i++){
            distances[i] = new double[map->getDimX()];
            parents[i] = new Point2<int>[map->getDimX()];
        }

        // function
        vector<Point2<int> > vp = minPathTwoPointsInternal(p0, p1, distances, parents);

        //delete
        for(int i=0; i<map->getDimY(); i++){
            delete [] distances[i];
            delete [] parents[i];
        }
        delete[] distances;
        delete[] parents;

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
                    if( 0<=ii && 0<=jj && ii<map->getDimY() && jj<map->getDimX() ){

                        if(map->getCellType(ii, jj) != OBST && (dist<initialDistAllowed || map->getCellType(ii, jj) != BODA )){ 
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
        static int counter = 1;
        cout << "\nCall of minPathTwoPointsInternalAngles, search for vict: " << counter++ << endl;

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
        if(equal( initialDir, baseDir, 0.001)){
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
        const int r = range; //range from class variable (default=3)
        const int side = 2*r+1;
        double computedDistances[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
        double computedAngles[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
        
        // cout << "\ncomputedAngles\n\t";
        // for(int j=(-r); j<=r; j++) 
        //     cout << j << "\t";
        for(int i=r; i>=(-r); i--){
            // cout << "\n" << i << ":\t";
            for(int j=(-r); j<=r; j++){

                computedDistances[(i+r)*side + (j+r)]  = sqrt( pow(i,2) + pow(j,2) );
                computedAngles[(i+r)*side + (j+r)] = (Point2<int>(0, 0).th( Point2<int>(j, i) )).toRad();
                
                // cout << (int)(computedAngles[(i+r)*side + (j+r)]*180.0/M_PI) << "\t";
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
                        if( 0<=ii && 0<=jj && ii<map->getDimY() && jj<map->getDimX() ){
                            // cell is allowed
                            if(map->getCellType(ii, jj) != OBST && (dist<initialDistAllowed || map->getCellType(ii, jj) != BODA )){ 

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
            cout << "The final distance for the point: " << endP <<  " from the point " << startP << endl;
            cout << "AKA the cells: " << endC << " and " << startC << endl;

            for(int q=0; q<nAngles; q++){
                double finalDist = distances[endC.y()][endC.x()][q];
                cout << q << ": Dir [" << q*45 << ", " << q*45+44 << "]: " << finalDist << endl;
                if( !equal(finalDist, baseDistance, 0.0001) && bestDist>finalDist ){
                    bestDist = finalDist;
                    bestDir = q;
                }
            }
            if(bestDir==-1){
                throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Something strange went wrong in the computation of the vector of parents in the minPath.", __LINE__, __FILE__);
            } else{
                cout << "Chosen the best! is " << bestDir << " with: " << bestDist << endl;
                cout << "endP: " << endP << endl;
                Point2<int> appP=endP;
                computedParents.resize(0);
                cout << "!!!\n";
                cout << "computedParents size: " << computedParents.size() << endl;
                cout << "!!!\n";
                // computing the vector of parents

                if (counter==4){
                    computedParents.push_back(Point2<int>(20,37));
                }
                else {
                    computedParents.push_back(endP);
                }
                cout << "???\n" << flush;
                int * v = new int[3] { endC.y(), endC.x(), bestDir };
                cout << "prima while\n";
                while(!( v[0]==startC.y() && v[1]==startC.x() )){
                    cout << "v[0]: " << v[0] << ", v[01]: " << v[1] << ", v[2]: " << v[2] << endl;
                    v = parents[v[0]][v[1]][v[2]];

                    // conversion from cell of the grid to point of the system (map)
                    computedParents.push_back( Point2<int>( (v[1]+0.5)*map->getPixX(), (v[0]+0.5)*map->getPixY() ));
                }
                reverse(computedParents.begin(), computedParents.end()); // I apply the inverse to have the vector from the begin to the end.
                cout << "deleting" << endl;
                delete[] v;
            }
            cout << "end" << endl;
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
    }

    /*! \brief It reset, to the given value, the matrix of distances, to compute again the minPath search.

        \param[in] value The value to be set.
    */
    void resetDistanceMap(double ** distances, const double value){
        for(int i=0; i<map->getDimY(); i++){
            for(int j=0; j<map->getDimX(); j++){
                distances[i][j] = value;
            }
        }
    }

    /*! \brief It reset, to the given value, the matrix of distances, to compute again the minPath search.

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
    void fromVpToPath(vector<Point2<int> > & vp, Path & path){
        if(vp.size()>=2){
            path.resize(0);
            Pose pose;
            double s=0.0; // k will be computed thanks to the dubins
            vp[0].invert();
            uint i=0;

            for(i=0; i<vp.size()-1; i++){

                vp[i+1].invert();
                double th = vp[i].th( vp[i+1] ).toRad();
                double k=0.1;

                if (equal(th, 0.01)){ 
                    k=0;
                }
                else if (th>M_PI){
                    k=-k;
                }

                pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, th, k);
                path.add(pose);

                s += vp[i].distance(vp[i+1])/SCALE;
            }
            pose = Pose( s, vp[i].x()/SCALE, vp[i].y()/SCALE, M_PI, 0.0);//use last th and k or simply 0????
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
    #define ROB_KMAX KMAX //0.01

    template <class T>
    vector<Point2<T> > reduce_points(Tuple<Point2<T> > init_points,
                                     Configuration2<double>* start=nullptr,
                                     Configuration2<double>* end=nullptr,
                                     int start_pos=0,
                                     int end_pos=-1){

        // vector<Point2<T> > v={Point2<T>(start.point().x(), start.point().y())};
        // //Compute a vector of points which are not on a line
        // for (int i=pos; i<init_points.size(); i++){
        //     Point2<T> app=init_points.get(i);
        //     if (i==0 || i==init_points.size()-1){
        //         v.push_back(app);
        //     }
        //     else {
        //         if (v.back().th(app).toRad()>DELTA){
        //             v.push_back(app);
        //         }
        //     }
        // }
        // for (int i=0; i<v.size()-1; i++){
        //     double th0=0.0, th1=0.0;
        //     if (i==0){
        //         th0=start.angle().toRad();
        //     }
        //     else {
        //         th1
        //     }
        //     Dubins<double> app=Dubins()
        // }

        // return ret;
    }

    template<class T>
    bool check_dubins_T (Tuple<Tuple<Point2<T> > >& vPDub){
        bool ok=true;
        for (int j=0; (j<3 && ok); j++){    
            for (int k=0; (k<(vPDub[j].size()-1) && ok); k++){    
                if (Planning::map->checkSegment(vPDub[j][k], vPDub[j][k+1])){
                    cout << "Segment through obstacles" << endl;
                    ok=false;
                }
                else if(!(vPDub[j][k].x()<Planning::map->getActualLengthX() && vPDub[j][k].y()<Planning::map->getActualLengthY() 
                        && vPDub[j][k].x()>Planning::map->getOffsetValue() && vPDub[j][k].y()>Planning::map->getOffsetValue())){
                    cout << "Point of index " << k << " out of map: " << vPDub[j][k] << endl;
                    ok=false;
                }
            }
            if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1])==OBJ_TYPE::OBST){
                cout << "Last point is on obstacle" << endl;
                ok=false;
            }
        }
        return ok;
    }

    //Maybe I'm never going to use this
    template<class T>
    bool check_dubins_V (vector<vector<Point2<T> > >& vPDub){
        bool ok=true;
        for (int j=0; (j<3 && ok); j++){    
            for (int k=0; (k<(vPDub[j].size()-1) && ok); k++){    
                if (Planning::map->checkSegment(vPDub[j][k], vPDub[j][k+1])){
                    cout << "Segment through obstacles" << endl;
                    ok=false;
                }
                else if(!(vPDub[j][k].x()<Planning::map->getActualLengthX() && vPDub[j][k].y()<Planning::map->getActualLengthY() 
                        && vPDub[j][k].x()>Planning::map->getOffsetValue() && vPDub[j][k].y()>Planning::map->getOffsetValue())){
                    cout << "Point of index " << k << " out of map: " << vPDub[j][k] << endl;
                    ok=false;
                }
            }
            if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1])==OBJ_TYPE::OBST){
                cout << "Last point is on obstacle" << endl;
                ok=false;
            }
        }
        return ok;
    }

    DubinsSet<double> create_dubins_path (  const vector<Point2<int> >& vP1,
                                            const vector<Point2<int> >& vP2,
                                            const Configuration2<double> endStart,
                                            uint pos)
    {
        DubinsSet<double> ret;
        //First I need to compute Dubins at extremities
        DubinsSet<double> best_intermediate;
        Tuple<Point2<double> > intermediates; intermediates.add(vP1.back()); //This cast should work without problems. BUT......
        uint i=0, j=0;
        for (i=pos; i<vP1.size()-1; i++){
            for (j=1; j<vP2.size()-1; j++){
                //Compute possible DubinsSet
                DubinsSet<double> app; 
                if (i==pos){ //If i==pos than I'm starting from a given configuration to a point in vP2
                    app=DubinsSet<double> (endStart, Configuration2<double>(vP2[j], vP2[j].th(vP2[j+1])), intermediates, ROB_KMAX); 
                }
                else { //Else I'm starting from an "unknown" point of vP1 to a point in vP2.
                    app=DubinsSet<double> (Configuration2<double>(vP1[i], vP1[i].th(vP1[i+1])), Configuration2<double>(vP2[j], vP2[j].th(vP2[j+1])), intermediates, ROB_KMAX);
                }

                if (app.getSize()>0){ //First I check DubinsSet was successful
                    //Then I split the Dubins and check if it's ok
                    Tuple<Tuple<Tuple<Point2<double> > > > dub_points = app.splitIt();
                    bool ok=true;

                    for (uint a=0; a<dub_points.size() && ok; a++){ //For each Dubins I check that's ok
                        ok=check_dubins_T(dub_points[a]);
                    }

                    if (!ok){
                        cout << "Dubins from point " << dub_points.front().front().front() << " to point " << dub_points.back().back().back() << " is not feasible" << endl;
                    }
                    else {
                        if (best_intermediate.getLength()>app.getLength()){
                            cout << "New best found " << best_intermediate.getLength() << ">" << app.getLength() << endl;
                            best_intermediate=app; //TODO this probably won't work until you implement the clone
                        }
                        else{
                            cout << "New best NOT found " << best_intermediate.getLength() << "<" << app.getLength() << endl;
                        }
                    }
                }

            }
        }
        if (i!=pos){ //Then there are some points from a DubinsSet to a Dubins that are not included.
            //Check if a line is feasible.

        } 

    }

    // template<class T>
    // void start_end_pos( const Configuration2<double>& anchorPoint, 
    //                 vector<Point2<T> >& vPoints,
    //                 const bool start)
    // {
    //     cout << "K: " << ROB_KMAX << endl;
    //     Dubins<double> dub;
    //     bool ok=false;
    //     vector<Point2<T> > new_points;
    //     uint id=0;
    //     const uint size=vPoints.size();
    //     for (uint i=size-2; (i>0 && !ok); i--){ //I continue until the points are empty or until a feasible dubins is not found. 
    //         cout << i << endl;
    //         ok=true; //Need to reset this each loop
    //         new_points.clear();
    //         if (start){
    //             dub=Dubins<double>(anchorPoint, Configuration2<double>(vPoints[i], vPoints[i].th(vPoints[i+1])), ROB_KMAX);
    //         }
    //         else {
    //             dub=Dubins<double>(Configuration2<double>(vPoints[size-i-1], vPoints[size-i-1].th(vPoints[size-i])), anchorPoint, ROB_KMAX);
    //         }

    //         if(dub.getId()<0){
    //             ok = false;
    //             cout << "Couldn't compute Dubins" << endl;
    //         }
    //         else {
    //             Tuple<Tuple<Point2<double> > > vPDub=dub.splitIt(0, 20); 
    //             cout << "Dubins split in: " << vPDub[0].size()+vPDub[1].size()+vPDub[2].size() << endl; 
    //             for (int j=0; (j<3 && ok); j++){    
    //                 for (int k=0; (k<(vPDub[j].size()-1) && ok); k++){    
    //                     if (Planning::map->checkSegment(vPDub[j][k], vPDub[j][k+1])){
    //                         cout << "Segment through obstacles" << endl;
    //                         ok=false;
    //                     }
    //                     else if(start && !(vPDub[j][k].x()<Planning::map->getActualLengthX() && vPDub[j][k].y()<Planning::map->getActualLengthY() 
    //                             && vPDub[j][k].x()>Planning::map->getOffsetValue() && vPDub[j][k].y()>Planning::map->getOffsetValue())){
    //                         cout << "Point of index " << k << " out of map: " << vPDub[j][k] << endl;
    //                         ok=false;
    //                     }
    //                     else {
    //                         new_points.push_back(vPDub[j][k]);
    //                     }
    //                 }
    //                 if (Planning::map->getPointType(vPDub[j][vPDub[j].size()-1])==OBJ_TYPE::OBST){
    //                     cout << "Last point is on obstacle" << endl;
    //                     ok=false;
    //                 }
    //             }
    //             if (ok){
    //                 id=i;
    //             }
    //         }
    //     }
    //     if (ok){
    //         cout << dub << endl;
    //         cout << "new_points size: " << new_points.size() << endl;
    //         cout << "Size before: " << size << endl;
    //         if (start){
    //             for (uint i=id+1; i<size; i++){
    //                 new_points.push_back(vPoints[i]);
    //             }
    //             vPoints=new_points;                
    //         }
    //         else {
    //             vector<Point2<int> > app;
    //             for (uint i=0; i<size-1-id; i++){
    //                 app.push_back(vPoints[i]);
    //             }
    //             for (uint i=0; i<new_points.size(); i++){
    //                 app.push_back(new_points[i]);
    //             }
    //             vPoints=app;
    //         }
    //         cout << "Size after: " << size << endl;
    //     }
    //     else {
    //         throw MyException<string> (EXCEPTION_TYPE::GENERAL, "No feasible path was found", __LINE__, __FILE__);
    //     }
    // }

    template<class T>
    Dubins<double> start_end_pos( const Configuration2<double>& anchorPoint, 
                                vector<Point2<T> >& vPoints,
                                uint& pos,
                                const bool start)
    {
        Dubins<double> dub;
        bool ok=false;
        const uint size=vPoints.size();
        for (uint i=size-2; (i>0 && !ok); i--){ //I continue until the points are empty or until a feasible dubins is not found. 
            cout << i << endl;
            ok=true; //Need to reset this each loop
            if (start){
                dub=Dubins<double>(anchorPoint, Configuration2<double>(vPoints[i], vPoints[i].th(vPoints[i+1])), ROB_KMAX);
            }
            else {
                dub=Dubins<double>(Configuration2<double>(vPoints[size-i-1], vPoints[size-i-1].th(vPoints[size-i])), anchorPoint, ROB_KMAX);
            }

            if(dub.getId()<0){
                ok = false;
                cout << "Couldn't compute Dubins" << endl;
            }
            else {
                Tuple<Tuple<Point2<double> > > vPDub=dub.splitIt(0, 20); 
                cout << "Dubins split in: " << vPDub[0].size()+vPDub[1].size()+vPDub[2].size() << endl; 
                
                ok=check_dubins_T(vPDub);
                if (ok){
                    pos=i;
                }
            }
        }
        if (ok){
            return dub;
        }
        else {
            pos=0;
            throw MyException<string> (EXCEPTION_TYPE::GENERAL, "No feasible path was found.", __LINE__, __FILE__);
        }
    }

    // template<class T>
    void plan_best( const Configuration2<double>& _start,
                    vector<vector<Point2<int> > >& vvPoints)
    {
        //Create Dubins for first path.
        Dubins<double> start;
        uint pos;
        try{
            start_end_pos(_start, vvPoints[0], pos, true);
        } catch(Exception e){
            cout << e.what() << endl;
        }
        cout << "Starting Dubins found." << endl;




        //Create intermediate Dubins.

        //Create Dubins for final path.
        try{
            start_end_pos(Configuration2<double>(vvPoints.back().back(), Angle(M_PI, Angle::RAD)), vvPoints.back(), pos, false); //TODO find best implementation for final angle
        }
        catch (Exception e){
            cout << e.what() << endl;
        }
        cout << "Ending Dubins found." << endl;

    }

}



