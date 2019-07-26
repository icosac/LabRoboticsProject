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
    distances = new int*[dimY];
    parents = new Point2<int>*[dimY];
    for(int i=0; i<dimY; i++){
        map[i] = new OBJ_TYPE[dimX];
        for(int j=0; j<dimX; j++){
            map[i][j] = FREE;
        }
        // the initializtion is to -1
        distances[i] = new int[dimX];
        parents[i] = new Point2<int>[dimX];
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
set<pair<int, int> > Mapp::cellsFromSegment(const Point2<int> p0, const Point2<int> p1){
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

/*! \brief Given a vector obstacles it is added them to the map.
    \details This means that all the cells of the map that are partly cover from these obstacles will be set to its type. It is a wrapper function of addObject.

    \param[in] vvp It is the vector of vector of points (set of convex hull) that delimit the objects of interest.
    \param[in] type It is the type of the given object. Defined as a OBJ_TYPE.
*/
void Mapp::addObjects(const vector< vector< Point2<int> > > & vvp, const OBJ_TYPE type){
    for(unsigned int i=0; i<vvp.size(); i++){
        addObject(vvp[i], type);
    }
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

/*! \brief Given a point return the type (status) of the cell in the map that contain it.

    \param[in] p The point of which we want to know the informations.
    \returns The type (OBJ_TYPE) of the cell.
*/
OBJ_TYPE Mapp::getPointType(const Point2<int> p){
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
bool Mapp::checkSegmentCollisionWithType(const Point2<int> p0, const Point2<int> p1, const OBJ_TYPE type){
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
bool Mapp::checkSegment(const Point2<int> p0, const Point2<int> p1){
    return(checkSegmentCollisionWithType(p0, p1, OBST));
}

/*! \brief Given a couple of points the function compute the minimum path that connect them avoiding the intersection of OBST and BODA.
    \details The function is based on a Breadth-first search (BFS).

    \param[in] startP The source point.
    \param[in] endP The destination point.
    \param[in] reset A boolean that reset the parameters of the BFS, by default it is true (highly recomended).
    \returns It is a vector of points along the path (one for each cell of the grid of the map).
*/
vector<Point2<int> > Mapp::minPathTwoPoints(const Point2<int> startP, const Point2<int> endP, const bool reset){
    if(reset){
        // unneccessary at the first run if all is initializated to 0. and also (maybe) in other very rare case based on multiple minPath call.
        resetDistanceMap();
    }

    // P=point, C=cell
    Point2<int> startC(startP.x()/pixX, startP.y()/pixY), endC(endP.x()/pixX, endP.y()/pixY);
    queue<Point2<int> > toProcess;

    toProcess.push(startC);
    distances[startC.y()/*i=y()*/][startC.x()/*j=x()*/] = 0;
    parents[  startC.y()/*i=y()*/][startC.x()/*j=x()*/] = startC;
    bool found = false;

    while(!toProcess.empty() && !found){
        // for each cell(8) around the selected one
        Point2<int> cell = toProcess.front();
        toProcess.pop();
        int iC = cell.y(), jC = cell.x(); //i and j of the cell
        int dist = distances[iC][jC];
        {
            int iIn = max(iC-1, 0), iEnd = min(iC+1, dimY-1);
            int jIn = max(jC-1, 0), jEnd = min(jC+1, dimX-1);
            for(int i=iIn; i<=iEnd; i++){
                for(int j=jIn; j<=jEnd; j++){
                    if(map[i][j] != OBST && map[i][j] != BODA){ 
                        if(i==endC.y() && j==endC.x()){
                            found = true;
                        }
                        if(i!=j){ // I do not concider the cell itself
                            // if not visited or bigger distance (probably not possible in this breath first search BFS)
                            if(distances[i][j]==baseDistance || distances[i][j] > dist+1){ 
                                distances[i][j] = dist+1;
                                parents[i][j] = cell;
                                //parents[i][j] = ( 3*(i-iC) + (j-jC) )*(-1) //the *(-1) is for the inversion: to refer the parent (cell) respect to the destination cell, and not vice versa

                                toProcess.push(Point2<int>(j, i));
                            }
                        }
                    }
                }
            }
        }
    }
    //todo gestire il caso in cui la destinazione non viene ragggiunta

    // reconstruct the vector of parents of the cells in the minPath
    vector<Point2<int> > computedParents;
    computedParents.push_back(endC);
    Point2<int> p = endC;    
    do {
        p = parents[p.y()][p.x()];

        // conversion from cell of the grid to point of the system (map)
        computedParents.push_back( Point2<int>(p.x()*pixX + pixX/2, p.y()*pixY + pixY/2) );
    } while( p==startC );
    reverse(computedParents.begin(), computedParents.end()); // I apply the inverse to have the vector from the begin to the end.

    return(computedParents);
}

/*! \brief It reset, to the given value, the matrix of distances, to compute again the minPath search.

    \param[in] value The value to be set.
*/
void Mapp::resetDistanceMap(const int value){
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            distances[i][j] = value;
        }
    }
}

/*! \brief It extracts from the given vector of points, a subset of points that always contains the first one and the last one.

    \param[in] n The number of points to select exept the extremes.
    \param[in] points The vector of points to be selected.
    \returns The vector containing the subset of n+2(begin and end) points.
*/
vector<Point2<int> > Mapp::sampleNPoints(const int n, const vector<Point2<int> > & points){
    vector<Point2<int> > vp;
    if(points.size() >= 2){
        int step = points.size()/(n+1);
        for(unsigned int i=0; i<points.size()-1; i+=step){
            vp.push_back(points[i]);
        }
        vp.push_back(points.back());
    }
    return(vp);    
}

/*! \brief The function create an image (Mat) with the dimensions of the Mapp and all its objects inside.
    \returns The generated image is returned.
*/
Mat Mapp::createMapRepresentation(){
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