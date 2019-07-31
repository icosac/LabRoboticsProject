#include <map.hh>

/*! \brief Constructor of the class.

    \param[in] _lengthX It is the size in pixel of the horizontal dimension.
    \param[in] _lengthY It is the size in pixel of the vertical dimension.
    \param[in] _pixX It is the horizontal granularity of a cell (how many pixels for each cell).
    \param[in] _pixY It is the vertical granularity of a cell (how many pixels for each cell).
    \param[in] _borderSize It is the dimension (defined based on cells of the map) of the border of each obstascles.
    \param[in] vvp It is a vector, of vector, of point that delimit, as a convex hull, a set of obstacles in the map.
*/
Mapp::Mapp( const int _lengthX, const int _lengthY, const int _pixX, const int _pixY, const int _borderSize, 
            const vector< vector<Point2<int> > > & vvp)
{
    lengthX = _lengthX;
    lengthY = _lengthY;
    pixX = _pixX;
    pixY = _pixY;
    borderSize = _borderSize;
    
    dimX = int(ceil(lengthX*1.0 / pixX));
    dimY = int(ceil(lengthY*1.0 / pixY));
    map = new OBJ_TYPE*[dimY];
    for(int i=0; i<dimY; i++){
        map[i] = new OBJ_TYPE[dimX];
        for(int j=0; j<dimX; j++){
            map[i][j] = FREE;
        }
    }

    for(unsigned int i=0; i<vvp.size(); i++){
        addObject(vvp[i], OBST);
    }
}

/*! \brief Given a segment (from p0 to p1) it return a set of all the cells that are partly cover from that segment.

    \param[in] p0 First point of the segment.
    \param[in] p1 Second point of the segment.
    \returns A set containing all the cells, identified by their row(i or y) and column(j or x).
*/
set<pair<int, int> > Mapp::cellsFromSegment(const Point2<int> & p0, const Point2<int> & p1){
    set<pair<int, int> > cells;
    
    // save in x0,y0 the point with the lowest x
    int x0, x1;
    double y0, y1;
    if( p0.x() < p1.x() ){
        x0=p0.x();  x1=p1.x();
        y0=p0.y();  y1=p1.y();
    } else{
        x0=p1.x();  x1=p0.x();
        y0=p1.y();  y1=p0.y();
    }

    double th = atan2(y1-y0, x1-x0);
    int j = x0/pixX;
    int i = int(y0)/pixY;

    // iteration over all the cells interssed from the side
    while(!(x0==x1 && equal(y0, y1, 0.0001))){
        //localize the left point on the grid

        cells.insert(pair<int, int>(i, j));

        // compute the y given the x
        int x = min(pixX*(j+1), x1);
        
        double l = (x-x0)*1.0/cos(th);
        double y;
        // to manage the situation of a vertical line
        if(int(l)==0){
            y = y1;
        } else{
            y = y0 + l*sin(th);
        }
        
        // color the whole line between the (j, i (aka i0)) and (j, i1)
        int i1 = int(floor(y))/pixY;
        for(int k=min(i, i1); k<=max(i, i1); k++){
            cells.insert(pair<int, int>(k, j));
        }

        x0 = x;
        y0 = y;
        j++;
        i = i1;
    }
    return(cells);
}

/*! \brief Given an obstacle it is added to the map.
    \details This means that all the cells of the map that are partly cover from this obstacle will be set to its type.

    \param[in] vp It is the vector of points (convex hull) that delimit the object of interest.
    \param[in] type It is the type of the given object. Defined as a OBJ_TYPE.
*/
void Mapp::addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type){
    // consistency check
    if(vp.size()<3){
        cout << "Invalid object less than 3 sides!\n";
    } else{
        //min, max of y computed (need for min&max for each line)
        int yMin=lengthY, yMax=0;
        for(unsigned int a = 0; a<vp.size(); a++){
            yMin = min(yMin, vp[a].y());
            yMax = max(yMax, vp[a].y());
        }
        
        // convert (min, max of y) to (min, max of i)
        int iMin=yMin/pixY, iMax=yMax/pixY;

        // generate vectors of min&max x for each line
        int vectSize = iMax-iMin+1;
        int * minimums = new int[vectSize];
        int * maximums = new int[vectSize];
        // vector<int> _maximums(vectSize); // fantastic example of a bug of std::vector

        for(int a=0; a<vectSize; a++){
            minimums[a] = dimX;
            maximums[a] = 0;
            // _maximums[a]=0;
        }

        // for each side of the object
        for(unsigned int s=0; s<vp.size(); s++){

            //save the end points of the segment
            set<pair<int, int> > collisionSet = cellsFromSegment(vp[s], vp[(s+1)%vp.size()]); //I use the module to consider all the n sides of an object with n points
            for(pair<int, int> el : collisionSet){

                int i=get<0>(el), j=el.second;  // two methods for get elements from a pair structure( get<0>(el)==el.first )
                map[i][j] = ((type==OBST) ? BODA : type);

                minimums[i-iMin] = min(minimums[i-iMin], j);
                maximums[i-iMin] = max(maximums[i-iMin], j);
                // _maximums[i-iMin] = max(_maximums[i-iMin], j);
            }
        }
        //draw the boder
        for(int k=0; k<vectSize; k++){
            for(int j=minimums[k]+1; j<maximums[k]; j++){
                if(type==OBST){
                    if( k<borderSize || vectSize-(k+1)<borderSize ||
                        j-minimums[k]<borderSize || maximums[k]-(j+1)<borderSize ){
                        map[k+iMin][j] = BODA;
                    } else{
                        map[k+iMin][j] = OBST;
                    }
                } else{
                    map[k+iMin][j] = type;
                }
            }
        }
        // delete[] minimums;
        // delete[] maximums; //NEVER uncomment this line. it cause "double free or corruption (out)" error ! ! ! 
    }
}

/*! \brief Given a vector objects it is added them to the map.
    \details This means that all the cells of the map that are partly cover from these obstacles will be set to its type. It is a wrapper function of addObject.

    \param[in] vvp It is the vector of vector of points (set of convex hull) that delimit the objects of interest.
    \param[in] type It is the type of the given object. Defined as a OBJ_TYPE.
*/
void Mapp::addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type){
    for(unsigned int i=0; i<vvp.size(); i++){
        addObject(vvp[i], type);
    }
}

/*! \brief Given a vector of obstacles adds them to the map.
    \details This means that all the cells of the map that are partly cover from these obstacles will be set to its type. It is a wrapper function of addObject.

    \param[in] objs It is the vector of obstacles to be loaded in the map structure.
*/
void Mapp::addObjects(const vector<Obstacle> & objs){
    for(auto el : objs){
        //enlarge the obstacle to allow to consider the robot as a point and not a circle.
        el.offsetting(offsetValue, lengthX-1, lengthY-1);

        vObstacles.push_back(el);
        addObject(el.getPoints(), OBST);
    }
}

/*! \brief Given a vector of gates (tipically this vector contain only one element) adds it to the map.
    \details This means that all the cells of the map that are partly cover from this gate will be set to its type. It is a wrapper function of addObject.

    \param[in] objs It is the vector of gates to be loaded in the map structure.
*/
void Mapp::addObjects(const vector<Gate> & objs){
    for(auto el : objs){
        vGates.push_back(el);
        addObject(el.getPoints(), GATE);
    }
}

/*! \brief Given a vector of victims adds them to the map.
    \details This means that all the cells of the map that are partly cover from these victims will be set to its type. It is a wrapper function of addObject.

    \param[in] objs It is the vector of victims to be loaded in the map structure.
*/
void Mapp::addObjects(const vector<Victim> & objs){
    for(auto el : objs){
        vVictims.push_back(el);
        addObject(el.getPoints(), VICT);
    }
}

/*! \brief Add to the given vector the set of centers of the victims of the map.
    \param[out] vp A vector where the requested centers will be added.
*/
void Mapp::getVictimCenters(vector<Point2<int> > & vp){
    for(auto el : vVictims){
        vp.push_back( el.getCenter() );
    }
}

/*! \brief Add to the given vector the center of the gate of the map.
    \param[out] vp A vector where the requested center will be added.
*/
void Mapp::getGateCenter(vector<Point2<int> > & vp){
    if(vGates.size()>=0){
        vp.push_back( vGates[0].getCenter() );
    }
}

/*! \brief Given a point return the type (status) of the cell in the map that contain it.

    \param[in] p The point of which we want to know the informations.
    \returns The type (OBJ_TYPE) of the cell.
*/
OBJ_TYPE Mapp::getPointType(const Point2<int> & p){
    int j = p.x()/pixX;
    int i = p.y()/pixY;

    return(map[i][j]);
}

/*! \brief Given a segment and a type, the function answer if that segment cross a cell with the given type.

    \param[in] p0 First point of the segment.
    \param[in] p1 Second point of the segment.
    \param[in] type The type to be detected.
    \returns True if the type was found, false otherwise.
*/
bool Mapp::checkSegmentCollisionWithType(const Point2<int> & p0, const Point2<int> & p1, const OBJ_TYPE type){
    set<pair<int, int> > collisionSet = cellsFromSegment(p0, p1);
    for(auto el:collisionSet){
        int i=get<0>(el), j=el.second;  // two methods for get elements from a pair structure
        if( map[i][j] == type ){
            return(true);
        }
    }
    return(false);
}

/*! \brief Given a segment, the function answer if that segment cross a cell with obstacles.
    \details It is a wrapper for the function 'checkSegmentCollisionWithType'.

    \param[in] p0 First point of the segment.
    \param[in] p1 Second point of the segment.
    \returns True if the obstacles were crossed, false otherwise.
*/
bool Mapp::checkSegment(const Point2<int> & p0, const Point2<int> & p1){
    return(checkSegmentCollisionWithType(p0, p1, OBST));
}

/*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
    \details The function is based on a Breadth-first search (BFS).

    \param[in] p0 The source point.
    \param[in] p1 The destination point.
    \returns A vector of vector of points along the path (one for each cell of the grid of the map). Each vector is the best path for one connection, given n points there are n-1 connecctions.
*/
vector<vector<Point2<int> > > Mapp::minPathNPoints(const vector<Point2<int> > & vp){
    //allocate
    double ** distances = new double*[dimY];
    Point2<int> ** parents = new Point2<int>*[dimY];
    for(int i=0; i<dimY; i++){
        // the initializtion is to -1
        distances[i] = new double[dimX];
        parents[i] = new Point2<int>[dimX];
    }

    //function
    vector<vector<Point2<int> > > vvp;
    for(unsigned int i=0; i<vp.size()-1; i++){
        resetDistanceMap(distances);
        vvp.push_back( minPathTwoPointsInternal(vp[i], vp[i+1], distances, parents) );
    }

    //delete
    for(int i=0; i<dimY; i++){
        delete [] distances[i];
        delete [] parents[i];
    }
    delete[] distances;
    delete[] parents;

    return(vvp);
}

/*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
    \details The function is based on a Breadth-first search (BFS).

    \param[in] p0 The source point.
    \param[in] p1 The destination point.
    \returns A vector of points along the path (one for each cell of the grid of the map).
*/
vector<Point2<int> > Mapp::minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1){
    //allocate
    double ** distances = new double*[dimY];
    Point2<int> ** parents = new Point2<int>*[dimY];
    for(int i=0; i<dimY; i++){
        distances[i] = new double[dimX];
        parents[i] = new Point2<int>[dimX];
    }

    // function
    resetDistanceMap(distances);
    vector<Point2<int> > tmp = minPathTwoPointsInternal(p0, p1, distances, parents);

    //delete
    for(int i=0; i<dimY; i++){
        delete [] distances[i];
        delete [] parents[i];
    }
    delete[] distances;
    delete[] parents;

    return( tmp );
}

/*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
    \details The function is based on a Breadth-first search (BFS).

    \param[in] startP The source point.
    \param[in] endP The destination point.
    \param[in] distances A matrix that is needed to store the distances of the visited cells.
    \param[in] parents A matrix that is needed to store the parent of each cell (AKA the one that have discovered that cell with the minimum distance).
    \returns A vector of points along the path (one for each cell of the grid of the map).
*/
vector<Point2<int> > Mapp::minPathTwoPointsInternal(
                        const Point2<int> & startP, const Point2<int> & endP, 
                        double ** distances, Point2<int> ** parents)
{
    // P=point, C=cell
    Point2<int> startC(startP.x()/pixX, startP.y()/pixY), endC(endP.x()/pixX, endP.y()/pixY);
    queue<Point2<int> > toProcess;
    // initialization of BFS
    toProcess.push(startC);
    distances[startC.y()/*i=y()*/][startC.x()/*j=x()*/] = 0.0;
    parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/] = startC;
    int found = 0;

    // precompute the computation of the distances inn the square of edges around the cell of interest
    const int r = range; //range from class variable (default=3)
    const int side = 2*r+1;
    double computedDistances[(int)pow(side, 2)]; // all the cells in a sqare of side where the center is the cell of interest
    for(int i=(-r); i<=r; i++){
        for(int j=(-r); j<=r; j++){
            computedDistances[(i+r)*side + (j+r)] = sqrt( pow(i,2) + pow(j,2) );
        }
    }

    // start iteration of the BFS
    while   (  !toProcess.empty()
              && found<=foundLimit
            ){
        // for each cell from the queue
        Point2<int> cell = toProcess.front();
        toProcess.pop();

        int iC = cell.y(), jC = cell.x(); //i and j of the cell
        double dist = distances[iC][jC];
                
        // for each possible edge
        for(int i=(-r); i<=r; i++){
            for(int j=(-r); j<=r; j++){
                // i&j are relative coordinates, ii&jj are absolute coordinates
                int ii = i+iC, jj = j+jC;

                // The cell itself (when i=0 and j=0) is considered (here) but never added to the queue due to the logic of the BFS
                if(0<=ii && 0<=jj && ii<dimY && jj<dimX){
                    if(map[ii][jj] != OBST && map[ii][jj] != BODA){ 
                        if(ii==endC.y() && jj==endC.x()){
                            found++;
                        }
                        double myDist = computedDistances[(i+r)*side + (j+r)];
                        // if not visited or bigger distance
                        if( equal(distances[ii][jj], baseDistance, 0.001) || distances[ii][jj] > dist + myDist ){
                            distances[ii][jj] = dist + myDist;
                            parents[ii][jj] = cell;

                            toProcess.push(Point2<int>(jj, ii));
                        }
                    }
                }
            }
        }
    }

    // reconstruct the vector of parents of the cells in the minPath
    vector<Point2<int> > computedParents;

    if(found==0){
        cerr << "\n\n\t\tDestination of minPath not reached ! ! !\nSegment from: " << startP << " to " << endP << "\nNo solution exist ! ! !\n\n";
    } else {
        // computing the vector of parents
        computedParents.push_back(endP);
        Point2<int> p = endC;
        while( p!=startC ){
            p = parents[p.y()][p.x()];

            // conversion from cell of the grid to point of the system (map)
            computedParents.push_back( Point2<int>(p.x()*pixX + pixX/2, p.y()*pixY + pixY/2) );
        }
        reverse(computedParents.begin(), computedParents.end()); // I apply the inverse to have the vector from the begin to the end.
    }
    return(computedParents);
}

/*! \brief It reset, to the given value, the matrix of distances, to compute again the minPath search.

    \param[in] value The value to be set.
*/
void Mapp::resetDistanceMap(double ** distances, const double value){
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            distances[i][j] = value;
        }
    }
}

/*! \brief It extracts from the given vector of vector of points, a subset of points that always contains the first one and the last one of each vector.

    \param[in] n The n number of points to sample.
    \param[in] points The vector of vector of points to be selected.
    \returns The vector containing the subset of n points.
*/
vector<Point2<int> > Mapp::sampleNPoints(const vector<vector<Point2<int> > > & vvp, const int n){
    vector<Point2<int> > vp;
    if(n < (int)vvp.size()+1){
        cout << "\n\nSampling N points: N is too small (at least vvp.size()+1 is required). . .\n\n";
    } else{
        int totalSize = 0;
        for(auto el : vvp){
            totalSize += el.size()-1;
        }
        float step = (totalSize-1)*1.0/(n-2);

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
        vp.push_back( vvp.back().back() );
    }
    return(vp);
}

/*! \brief It extracts from the given vector of points, a subset of points that always contains the first one and the last one.

    \param[in] n The number of points to select exept the extremes, it must be greater or equal than 2.
    \param[in] points The vector of points to be selected.
    \returns The vector containing the subset of n points.
*/
vector<Point2<int> > Mapp::sampleNPoints(const vector<Point2<int> > & points, const int n){
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
vector<Point2<int> > Mapp::samplePointsEachNCells(const vector<Point2<int> > & points, const int step){
    vector<Point2<int> > vp;
    if(step<=1 || points.size()==2){
        vp = points;
    } else if (points.size() > 2){
        for(unsigned int i=0; i<points.size()-1; i+=step){
            vp.push_back(points[ i ]);
        }
        vp.push_back(points.back());
    } else{
        cout << "Invalid value of step and dimension of the vector.\n\n";
    }
    return(vp);    
}
/*! \brief The function create an image (Mat) with the dimensions of the Mapp and all its objects inside.
    \returns The generated image is returned.
*/
Mat Mapp::createMapRepresentation(){
    // example code at: https://docs.opencv.org/2.4/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html
    Mat imageMap(lengthY, lengthX, CV_8UC3, Scalar(47, 98, 145));// empty map
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            if(map[i][j]!=FREE){
                // choose the color according to the type
                Scalar color;
                switch (map[i][j]){
                    case OBST:
                        color = Scalar(0, 0, 255); //BGR format
                        break;
                    case BODA:
                        color = Scalar(0, 0, 50); //BGR format
                        break;
                    case VICT:
                        color = Scalar(0, 255, 0); //BGR format
                        break;
                    case GATE:
                        color = Scalar(255, 0, 0); //BGR format
                        break;
                    default:
                        break;
                }
                // color the relative rectangle
                rectangle(imageMap, Point(j*pixX, i*pixY), Point((j+1)*pixX, (i+1)*pixY), color, -1);
            }
        }
    }
    return(imageMap);
}

/*! \brief It add to the given image a set of (n-1) segments specified by the n points given.

    \param[in/out] map The image where the segments will be added.
    \param[in] vp The vector of points that identify the segments.
    \param[in] thickness The thickness of the lines to be drawn.
*/
void Mapp::imageAddSegments(Mat & image, const vector<Point2<int> > & vp, const int thickness){
    for(unsigned int i=0; i<vp.size()-1; i++){
        line( image, 
            Point(vp[ i ].x(), vp[ i ].y()), 
            Point(vp[i+1].x(), vp[i+1].y()), 
            Scalar(0, 255, 255),
            thickness);
    }
}

/*! \brief It add to the given image the segment defined from p0 to p1.

    \param[in/out] map The image where the segment will be added.
    \param[in] p0 The first point of the segment.
    \param[in] p1 The end point of the segment.
    \param[in] thickness The thickness of the line to be drawn.
*/
void Mapp::imageAddSegment(Mat & image, const Point2<int> & p0, const Point2<int> & p1, const int thickness){
    line( image, 
        Point(p0.x(), p0.y()), 
        Point(p1.x(), p1.y()), 
        Scalar(0, 255, 255),
        thickness);
}

/*! \brief It add to the given image a vector of points.

    \param[in/out] map The image where the point will be added.
    \param[in] vp The vecotor of points to add.
    \param[in] radius The radius of the points to be drawn.
*/
void Mapp::imageAddPoints(Mat & image, const vector<Point2<int> > & vp, const int radius){
    for(Point2<int> el : vp){
        circle(image, Point(el.x(), el.y()), radius, Scalar(0, 255, 255), -1);
    }
}

/*! \brief It add to the given image a point.

    \param[in/out] map The image where the points will be added.
    \param[in] p The point to add.
    \param[in] radius The radius of the point to be drawn.
*/
void Mapp::imageAddPoint(Mat & image, const Point2<int> & p, const int radius){
    circle(image, Point(p.x(), p.y()), radius, Scalar(0, 255, 255), -1);
}

/*! \brief Print to the terminal the main informations of the Map.
*/
void Mapp::printDimensions(){
    cout << "lengthX: " << lengthX << ", lengthY: " << lengthY << endl;
    cout << "dimX: " << dimX << ", dimY: " << dimY << endl;
    cout << "pixX: " << pixX << ", pixY: " << pixY << endl;
}

/*! \brief Generate a string (a grid of pixels) that represent the matrix.
    \returns The generated string. 
*/
string Mapp::matrixToString(){
    stringstream ss;
    
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            ss << map[i][j] << "";
        }
        ss << endl;
    }
    ss << endl;

    string s = ss.str();
    return(s);
}

/*! \brief Print to the terminal the main informations of the Map, and its grid representation.
*/
void Mapp::printMap(){
    printDimensions();
    cout << matrixToString();
}