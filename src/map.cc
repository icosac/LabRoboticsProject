#include <map.hh>

/*! \brief Constructor of the class.

    \param[in] _lengthX It is the size in pixel of the horizontal dimension.
    \param[in] _lengthY It is the size in pixel of the vertical dimension.
    \param[in] _pixX It is the horizontal granularity of a cell (how many pixels for each cell).
    \param[in] _pixY It is the vertical granularity of a cell (how many pixels for each cell).
    \param[in] vvp It is a vector, of vector, of point that delimit, as a convex hull, a set of obstacles in the map.
*/
Mapp::Mapp(int _lengthX, int _lengthY, int _pixX, int _pixY, vector< vector<Point2<int> > > vvp){
    lengthX = _lengthX;
    lengthY = _lengthY;
    pixX = _pixX;
    pixY = _pixY;
    
    dimX = int(ceil(lengthX*1.0 / pixX));
    dimY = int(ceil(lengthY*1.0 / pixY));
    map = new OBJ_TYPE*[dimY];
    for(int i=0; i<dimY; i++){
        map[i] = new OBJ_TYPE[dimX];
    }

    for(uint i=0; i<vvp.size(); i++){
        addObject(vvp[i], OBST);
    }
}

/*! \brief Given a segment (from p0 to p1) it return a set of all the cells that are partly cover from that segment.

    \param[in] p0 First point of the segment.
    \param[in] p1 Second point of the segment.
    \returns A set containing all the cells, identified by their row(i or y) and column(j or x).
*/
set<pair<int, int> > Mapp::cellsFromSegment(Point2<int> p0, Point2<int> p1){
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

/*! \brief Given an obstacle it is added to the map.
    \details This means that all the cells of the map that are partly cover from this obstacle will be set to its type.

    \param[in] vp It is the vector of points (convex hull) that delimit the object of interest.
    \param[in] type It id the type of the given object. Defined as a OBJ_TYPE.
*/
void Mapp::addObject(vector<Point2<int> > vp, const OBJ_TYPE type){
    if(vp.size()>=3){
        //cout << "Points:\n";
        //min, max of x, y computed
        int yMin=lengthY, yMax=0;
        for(uint a = 0; a<vp.size(); a++){
            //cout << "\t" << vp[a] << endl;
            yMin = min(yMin, vp[a].y());
            yMax = max(yMax, vp[a].y());
        }
        //cout << endl;
        
        int iMin=yMin/pixY, iMax=yMax/pixY;
        //cout << "yMin: " << yMin << ", yMax: " << yMax << endl;
        //cout << "iMin: " << iMin << ", iMax: " << iMax << endl;

        int vectSize = iMax-iMin+1;
        vector<int> minimums;
        vector<int> maximums;
        for(int a=0; a<vectSize; a++){
            minimums.push_back(dimX);
            maximums.push_back(0);
        }

        vp.push_back(vp[0]);
        // for each side of the object
        for(uint s=0; s<vp.size()-1; s++){
            // cout << "\n_______________________________________________________\nCouple n:" << s+1 << endl;
            //save the end points of the segment
            set<pair<int, int> > collisionSet = cellsFromSegment(vp[s], vp[s+1]);
            // cout << "Set returned:\n";
            // cout << "size " << collisionSet.size() << endl;
            for(auto el:collisionSet){
                int i=get<0>(el), j=el.second;  // two methods for get elements from a pair structure
                // cout << j << "," << i << " - ";
                map[i][j] = type;
                minimums[i-iMin] = min(minimums[i-iMin], j);
                maximums[i-iMin] = max(maximums[i-iMin], j);
            }
            // cout << endl;
        }
        for(int k=0; k<vectSize; k++){
            //cout << "line " << k+iMin << ": (" << minimums[k] << ", " << maximums[k] << ")\n";
            for(int j=minimums[k]+1; j<maximums[k]; j++){
                map[k+iMin][j] = type;
            }
        }
    } else{
        cout << "Invalid object less than 3 sides!\n";
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