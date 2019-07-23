#include <map.hh>

void out(int c){cout << c << endl;}

/*! \brief Constructor of the class.

    \param[in] _lengthX It is the size in pixel of the horizontal dimension.
    \param[in] _lengthY It is the size in pixel of the vertical dimension.
    \param[in] _pixX It is the horizontal granularity of a cell (how many pixels for each cell).
    \param[in] _pixY It is the vertical granularity of a cell (how many pixels for each cell).
    \param[in] vvp It is a vector, of vector, of point that delimit, as a convex hull, a set of obstacles in the map.
*/
Mapp::Mapp( const int _lengthX, const int _lengthY, const int _pixX, const int _pixY, const int _borderSize, 
            const vector< vector<Point2<int> > > & vvp){
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
    //cout << x0 << "," << y0 << " - " << x1 << "," << y1 << endl;
    double th = atan2(y1-y0, x1-x0);

    int j = x0/pixX;
    int i = int(y0)/pixY;
    //cout << "i: " << i << ", j: " << j << endl;

    // iteration over all the cells interssed from the side
    while(!(x0==x1 && equal(y0, y1, 0.0001))){
        //cout << "check x: " << (x0==x1) << ", check y: " << equal(y0, y1) << endl;
        //cout << "check fail on -> " << x0 << "," << y0 << " - " << x1 << "," << y1 << endl;
        //localize the left point on the grid
        //cout << "adding pixel: " << i << ", " << j << endl;

        cells.insert(pair<int, int>(i, j));

        // compute the y given the x
        int x = min(pixX*(j+1), x1);
        //cout << "x: " << x;
        
        double l = (x-x0)*1.0/cos(th);
        double y;
        // to manage the situation of a vertical line
        if(int(l)==0){
            y = y1;
        } else{
            y = y0 + l*sin(th);
        }
        //cout << ", y: " << y << endl;
        
        // color the whole line between the (j, i (aka i0)) and (j, i1)
        int i1 = int(floor(y))/pixY;
        //cout << "i: " << i << " i1: " << i1 << endl;
        //cout << "for: \n";
        for(int k=min(i, i1); k<=max(i, i1); k++){
            //cout << "\t" << k << " " << j << endl;
            cells.insert(pair<int, int>(k, j));
        }
        //cout << endl;

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
        cout << "in\n";  out(i*11111);
        addObject(vvp[i], type);
        cout << "out\n"; out(i*11111);
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
        cout << "Points: (size" << vp.size() << ")\n";
        //min, max of y computed (need for min&max for each line)
        int yMin=lengthY, yMax=0;
        for(unsigned int a = 0; a<vp.size(); a++){
            cout << "\t" << vp[a] << endl;
            yMin = min(yMin, vp[a].y());
            yMax = max(yMax, vp[a].y());
        }
        cout << endl;
        
        // convert (min, max of y) to (min, max of i)
        int iMin=yMin/pixY, iMax=yMax/pixY;
        cout << "yMin: " << yMin << ", yMax: " << yMax << endl;
        cout << "iMin: " << iMin << ", iMax: " << iMax << endl;

        // generate vectors of min&max x for each line
        int vectSize = iMax-iMin+1;
        vector<int> minimums;
        vector<int> maximums;
        for(int a=0; a<vectSize; a++){
            minimums.push_back(dimX);
            maximums.push_back(0);
        }

        // for each side of the object
        for(unsigned int s=0; s<vp.size(); s++){
            cout << "\n_______________________________________________________\nCouple n:" << s+1 << endl;
            //save the end points of the segment
            set<pair<int, int> > collisionSet = cellsFromSegment(vp[s], vp[(s+1)%vp.size()]); //I use the module to consider all the n sides of an object with n points
            cout << "Set returned:\n";
            cout << "size " << collisionSet.size() << endl;
            for(pair<int, int> el:collisionSet){
                int i=el.first, j=el.second;  // two methods for get elements from a pair structure( get<0>(el)==el.first )
                //int i=70, j=95;
                cout << j << "," << i << " - ";
                map[i][j] = ((type==OBST) ? BODA : type);
                minimums[i-iMin] = min(minimums[i-iMin], j);
                cout << "aaaaa" << maximums.size() << "  " << i-iMin << "  " << j << endl;
                try{
                    maximums[i-iMin] = max(maximums[i-iMin], j);
                } catch(const exception& e) {
                    cout << "\n\n\n\n\n\n\n\n\t\t\tECCEZIONE GENERATA\n\n\n\n\n\n\n\n\n\n";
                }
            }
            cout << endl;
        }
        cout << "borderSize: " << borderSize << ", vectSize: " << vectSize << ", iMin: " << iMin << endl;
        for(int k=0; k<vectSize; k++){
            cout << "line " << k+iMin << ": (" << minimums[k] << ", " << maximums[k] << ")\n";
            for(int j=minimums[k]+1; j<maximums[k]; j++){
                //cout << "\tinside for line: " << k+iMin << endl;
                // TODO: test
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
        cout << "out of for\n";
    }
    cout << "OK" << endl;
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

void Mapp::resetDistanceMap(const int value){
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            distances[i][j] = value;
            //parents[i][j] = 0; //unneccessary reset (variable never read before the write)
        }
    }
}

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

// todo add in the .hh file
// vector< Configuration2<int> >/*is it correct???*/ Mapp::fromPointsToConfiguarations(const vector<Point2<int> > & vp){
//     vector< Configuration2<int> > vC;
//     for(unsigned int i=0; i<vp.size()-1; i++){
//         Angle th = //compute angle from vp[i], vp[i+1]
//         vC.push_back( Configuration )
//     }
// }


/*! \brief The function create an image (Mat) with the dimensions of the Mapp and all its objects inside.
    \returns The generated image is returned.
*/
Mat Mapp::createMapRepresentation(/*eventually add a vector of bubins*/){
    // empty map
	Mat imageMap = Mat::zeros( Size(lengthX, lengthY), CV_8UC3 );
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
                        color = Scalar(255, 0, 255); //BGR format
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
            ss << map[i][j] << " ";
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