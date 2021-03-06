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

/*! \brief Destructor of the class.
*/
Mapp::~Mapp(){
    for(int i=0; i<dimY; i++){
        delete [] map[i];
    }
    delete[] map;
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

/*! \brief Given an object it is added to the map.
    \details This means that all the cells of the map that are partly cover from this obstacle will be set to its type.

    \param[in] vp It is the vector of points (convex hull) that delimit the object of interest.
    \param[in] type It is the type of the given object. Defined as a OBJ_TYPE.
*/
void Mapp::addObject(const vector<Point2<int> > & vp, const OBJ_TYPE type){
    // consistency check
    if(vp.size()<3){
        throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Impossible to create the object, less than 3 sides.", __LINE__, __FILE__);
    } 
    else {
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
        int minimums[lengthY];
        int maximums[lengthY];

        for(int a=0; a<vectSize; a++){
            minimums[a] = dimX;
            maximums[a] = 0;
        }

        // for each side of the object
        for(unsigned int s=0; s<vp.size(); s++){

            //save the end points of the segment
            set<pair<int, int> > collisionSet = cellsFromSegment(vp[s], vp[(s+1)%vp.size()]); //I use the module to consider all the n sides of an object with n points
            for(pair<int, int> el : collisionSet){

                int i=get<0>(el), j=el.second;  // two methods for get elements from a pair structure( get<0>(el)==el.first )
                // map[i][j] = ((type==OBST) ? BODA : type);

                minimums[i-iMin] = min(minimums[i-iMin], j);
                maximums[i-iMin] = max(maximums[i-iMin], j);
            }
        }

        // draw the BODA and the inside of the objects
        for(int k=0; k<vectSize; k++){
            for(int j=minimums[k]; j<=maximums[k]; j++){
                
                if(map[k+iMin][j] != OBST){
                    map[k+iMin][j] = type;
                }
            }
        }

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

        // local modify for adding the border (BODA)
        el.offsetting(borderSizeDefault, lengthX-1, lengthY-1);
        addObject(el.getPoints(), BODA);
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
    if (!this->checkPointInMap(p)){
        return OUT_OF_MAP;
    }
    int j = p.x()/pixX;
    int i = p.y()/pixY;
    return(map[i][j]);
}

/*! \brief Given a cell return its type.

    \param[in] i The row of the cell.
    \param[in] j The column of the cell.
    \returns The type (OBJ_TYPE) of the requested cell.
*/
OBJ_TYPE Mapp::getCellType(const int i, const int j){
    if(!this->checkCellInMap(i, j)){
        return OUT_OF_MAP;
    }
    return(map[i][j]);
}

/*! \brief Given a segment and a type, the function answer if that segment cross a cell with the given type.

    \param[in] p0 First point of the segment.
    \param[in] p1 Second point of the segment.
    \param[in] type The type to be detected.
    \returns True if the type was found, false otherwise.
*/
bool Mapp::checkSegmentCollisionWithType(const Point2<int> & p0, const Point2<int> & p1, const OBJ_TYPE type){
    if (!this->checkPointInMap(p0) || !this->checkPointInMap(p1)){
        return false;
    }
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

/*! \brief Given a point, the function answer if that point is inside the map.

    \param[in] p The point that need to be checked.
    \returns True if the point is inside the map, false otherwise.
*/
bool Mapp::checkPointInMap(Point2<int> P) {
    return (P.x()<this->getLengthX() && P.y()<this->getLengthY() 
            && P.x()>=0 && P.y()>=0);
}


/*! \brief Given a point, the function answer if that point is inside the actual map. This means that the border of the map also consider the offsetting due to the robot and not only the point.

    \param[in] p The point that need to be checked.
    \returns True if the point is inside the actual map, false otherwise.
*/
bool Mapp::checkPointInActualMap(Point2<int> P){
    return (P.x()<this->getActualLengthX() && P.y()<this->getActualLengthY() 
            && P.x()>this->getOffsetValue() && P.y()>this->getOffsetValue());
}

/*! \brief Given a cell(defined with its row and column), the function answer if that cell is inside the cell representation of the map.

    \param[in] i The i=row of the cell.
    \param[in] j The j=column of the cell.
    \returns True if the cell is inside the cell representation of the map, false otherwise.
*/
bool Mapp::checkCellInMap(const int i, const int j){
    return (j>=0 && i>=0 &&
            j<this->getDimX() && i<this->getDimY());
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
                        throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Wrong type of the map, it is impossible to create the representation.", __LINE__, __FILE__);
                    break;
                }
                // color the relative rectangle
                rectangle(imageMap, Point(j*pixX, i*pixY), Point((j+1)*pixX, (i+1)*pixY), color, -1);
            }
        }
    }
    
    imageAddSegment(imageMap, Point2<int>(getOffsetValue(), getOffsetValue()), Point2<int>(getActualLengthX(), getOffsetValue()), 3, Scalar(0, 0, 50));
    imageAddSegment(imageMap, Point2<int>(getActualLengthX(), getOffsetValue()), Point2<int>(getActualLengthX(), getActualLengthY()), 3, Scalar(0, 0, 50));
    imageAddSegment(imageMap, Point2<int>(getActualLengthX(), getActualLengthY()), Point2<int>(getOffsetValue(), getActualLengthY()), 3, Scalar(0, 0, 50));
    imageAddSegment(imageMap, Point2<int>(getOffsetValue(), getActualLengthY()), Point2<int>(getOffsetValue(), getOffsetValue()), 3, Scalar(0, 0, 50));

    return(imageMap);
}

/*! \brief It add to the given image a set of (n-1) segments specified by the n points given.

    \param[in/out] map The image where the segments will be added.
    \param[in] v The vector of points that identify the segments.
    \param[in] thickness The thickness of the lines to be drawn.
*/
void Mapp::imageAddSegments(Mat & image, const vector<Point2<int> > & v, const int thickness){
    for(int i=0; i<(int)(v.size())-1; i++){
        line( image, 
            Point(v[ i ].x(), v[ i ].y()), 
            Point(v[i+1].x(), v[i+1].y()), 
            Scalar(0, 255, 255),
            thickness);
    }
}

/*! \brief It add to the given image a set of (n-1) segments specified by the n points given.

    \param[in/out] map The image where the segments will be added.
    \param[in] v The vector of points that identify the segments.
    \param[in] thickness The thickness of the lines to be drawn.
*/
void Mapp::imageAddSegments(Mat & image, const vector<Configuration2<double> > & v, const int thickness){
    for(int i=0; i<(int)(v.size())-1; i++){
        line( image, 
            Point((int)v[ i ].x(), (int)v[ i ].y()), 
            Point((int)v[i+1].x(), (int)v[i+1].y()), 
            Scalar(0, 255, 255),
            thickness);
    }
}

/*! \brief It add to the given image the segment defined from p0 to p1.

    \param[in/out] map The image where the segment will be added.
    \param[in] p0 The first point of the segment.
    \param[in] p1 The end point of the segment.
    \param[in] thickness The thickness of the line to be drawn.
    \param[in] colot The color of the line to be drawn.
*/
void Mapp::imageAddSegment(Mat & image, const Point2<int> & p0, const Point2<int> & p1, const int thickness, const Scalar color){
    line( image, 
        Point(p0.x(), p0.y()), 
        Point(p1.x(), p1.y()), 
        color,
        thickness);
}

/*! \brief It add to the given image a vector of points.

    \param[in/out] map The image where the point will be added.
    \param[in] v The vecotor of points to add.
    \param[in] radius The radius of the points to be drawn.
*/
void Mapp::imageAddPoints(Mat & image, const vector<Point2<int> > & v, const int radius){
    for(Point2<int> el : v){
        circle(image, Point(el.x(), el.y()), radius, Scalar(0, 255, 255), -1);
    }
}

/*! \brief It add to the given image a vector of points.

    \param[in/out] map The image where the point will be added.
    \param[in] v The vecotor of points to add.
    \param[in] radius The radius of the points to be drawn.
*/
void Mapp::imageAddPoints(Mat & image, const vector<Configuration2<double> > & v, const int radius){
    for(Configuration2<double> el : v){
        circle(image, Point((int)el.x(), (int)el.y()), radius, Scalar(0, 255, 255), -1);
    }
}

/*! \brief It add to the given image a point.

    \param[in/out] map The image where the points will be added.
    \param[in] p The point to add.
    \param[in] radius The radius of the point to be drawn.
    \param[in] color The color of the point to be drawn.
*/
void Mapp::imageAddPoint(Mat & image, const Point2<int> & p, const int radius, const Scalar color){
    circle(image, Point(p.x(), p.y()), radius, color, -1);
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