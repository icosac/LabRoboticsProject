#include "objects.hh"

//_______________________________   Object   _______________________________
/*! \brief Generate a string that describe the object.
    \returns The generated string.
*/
string Object::toString(){
    stringstream ss;
    ss << "center: [" << center << "], radius=" << radius << "\n";
    if(points.size()>0){
        ss << points.size() << " points:\n[";
        for(unsigned i=0; i<points.size(); i++){
            ss << "(" << points[i] << "), ";
        }
        ss << "\b\b]\n";
    } else{
        ss << "noone point\n";
    }
    string s = ss.str();
    return(s);
}

/*! \brief Return the number of points of the object.
    \returns The number of points.
*/
unsigned int Object::size(){
    return(points.size());
}

/*! \brief Return the number of points of the object.
    \returns The number of points.
*/
unsigned int Object::nPoints(){
    return(points.size());
}

/*! \brief Return the of points of the object.
    \returns The vector of points.
*/
vector<Point2<int> > Object::getPoints(){
    return(points);
}

/*! \brief Retrieve the center of the object.
    \returns The center.
*/
Point2<int> Object::getCenter(){
    return(center);
}

/*! \brief Retrieve the radius of the object.
    \returns The radius.
*/
double Object::getRadius(){
    return(radius);
}

/*! \brief Find the representative center of the object.
    \details The center is computed as the mean of the minimum and maximum x and y.
*/
void Object::computeCenter(){
    if(points.size()>=1){
        int minX=points[0].x(), maxX=points[0].x();
        int minY=points[0].y(), maxY=points[0].y();
        
        for(unsigned i=1; i<points.size(); i++){
            if(points[i].x()<minX){
                minX = points[i].x();
            }
            if(points[i].x()>maxX){
                maxX = points[i].x();
            }

            if(points[i].y()<minY){
                minY = points[i].y();
            }
            if(points[i].y()>maxY){
                maxY = points[i].y();
            }
        }
        center.x( (int)round((minX+maxX)/2.0) );
        center.y( (int)round((minY+maxY)/2.0) );
    } else{
        throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Not enough points for computing the centre.", __LINE__, __FILE__);
    }
}

/*! \brief Compute the radius of the object.
    \details This function assume that the center of the object is already computed and consistent.
*/
void Object::computeRadius(){
    if(points.size()>=1){
        double dist, maxRadius = center.distance(points[0]);
        for(unsigned i=1; i<points.size(); i++){
            dist = center.distance(points[i]);
            if(dist>maxRadius){
                maxRadius = dist;
            }
        }
        this->radius = maxRadius;
    } else{
        throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Not enough points for computing the radius.", __LINE__, __FILE__);
    }
}

/*! \brief Enlarge the object of the given offset (defined as pixels=mm in our scenario).
    \details The function automatically update even the center and the radius.

    \param[in] offset The size of the offset.
    \param[in] limitX The the maximum x that the point can have.
    \param[in] limitY The the maximum y that the point can have.
*/
void Object::offsetting(const int offset, const int limitX, const int limitY){

    //documentation: http://www.angusj.com/delphi/clipper.php
    ClipperLib::Path srcPoly; //A Path represents a polyline or a polygon
    ClipperLib::Paths solution;
    for(unsigned i=0; i<points.size(); i++){
        //Add the list of points to the Path object using operator <<
        srcPoly << ClipperLib::IntPoint(points[i].x(), points[i].y()); 
    }
    ClipperLib::ClipperOffset co;
    //A ClipperOffset object provides the methods to offset a given (open or closed) path
    co.AddPath(srcPoly, ClipperLib::jtSquare, ClipperLib::etClosedPolygon); 
    co.Execute(solution, offset);

    // save back the new polygon (convex)
    points.resize(0);
    for(unsigned i=0; i<solution.size(); i++){
        if( Orientation(solution[i]) ){ //returning true for outer polygons and false for hole polygons
            for(unsigned j=0; j<solution[i].size(); j++){
                // force the point to stay inside the map
                int x = max( min((int)solution[i][j].X, limitX), 0);
                int y = max( min((int)solution[i][j].Y, limitY), 0);
                points.push_back( Point2<int>(x, y) );
            }
        } else{
            throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Hole polygon found, it must be convex.", __LINE__, __FILE__);
        }
    }
    computeCenter();
    computeRadius();
}

/*! \brief Check if the given point is inside the approximation shape of the object (a circle).

    \param[in] pt The point to be checked.
    \returns True if the point is inside the object, false otherwise.
*/
bool Object::insidePolyApprox(Point2<int> pt){
    return((pt.distance(center)<=radius) ? true : false);
}

/*! \brief Exact check if a point is inside the object (no approximation).

    \param[in] pt The point to be checked.
    \returns True if the point is inside the object, false otherwise.
*/
bool Object::insidePoly(Point2<int> pt){
    if(points.size()<3){
        return(false);
    } else{
        //n>2 Keep track of cross product sign changes
        int pos = 0;
        int neg = 0;

        for(unsigned i=0; i<points.size(); i++){
            //If point is in the polygon
            if (points[i].x()==pt.x() && points[i].y()==pt.y())
                return true;

            //Form a segment between the i'th point
            int x1 = points[i].x();
            int y1 = points[i].y();

            //And the i+1'th, or if i is the last, with the first point
            int i2 = ((i<points.size()-1) ? i+1 : 0);

            int x2 = points[i2].x();
            int y2 = points[i2].y();

            int x = pt.x();
            int y = pt.y();

            //Compute the cross product
            int d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1);

            if(d > 0){
                pos++;
            }
            if(d < 0){
                neg++;
            }

            //If the sign changes, then point is outside
            if (pos>0 && neg>0){
                return(false);
            }
        }
    }
    //If no change in direction, then on same side of all segments, and thus inside
    return(true);
}

//_______________________________   Obstacle   _______________________________
/*! \brief Constructor of the obstacle class and automatically compute center and radius.

    \param[in] vp Vector of points that is the convex hull of the obstacle.
    \returns Return the created obstacle.
*/
Obstacle::Obstacle(vector<Point2<int> > & vp){
    points = vp;
    if(vp.size()==0){
        center = Point2<int>(-1, -1);
        radius = -1.0;
    } else{
        computeCenter();
        computeRadius();
    }
}

/*! \brief Generate a string that describe the obstacle.
    \returns The generated string.
*/
string Obstacle::toString(){
    return("Obstacle:\n" + Object::toString());
}

/*! \brief Print the describing string of the obstacle.
*/
void Obstacle::print(){
    cout << toString();
}

//_______________________________   Gate   _______________________________
/*! \brief Constructor of the gate class and automatically compute center and radius.

    \param[in] vp Vector of points that is the convex hull of the gate.
    \returns Return the created gate.
*/
Gate::Gate(vector<Point2<int> > & vp){
    points = vp;
    if(vp.size()==0){
        center = Point2<int>(-1, -1);
        radius = -1.0;
    } else{
        computeCenter();
        computeRadius();
    }
}

/*! \brief Generate a string that describe the gate.
    \returns The generated string.
*/
string Gate::toString(){
    return("Gate:\n" + Object::toString());
}

/*! \brief Print the describing string of the gate.
*/
void Gate::print(){
    cout << toString();
}

//_______________________________   Victim   _______________________________
/*! \brief Constructor of the victim class and automatically compute center and radius.

    \param[in] vp Vector of points that is the convex hull of the victim.
    \param[in] _value The representative number of the victim.
    \returns Return the created victim.
*/
Victim::Victim(vector<Point2<int> > & vp, int _value){
    points = vp;
    value = _value;
    if(vp.size()==0){
        center = Point2<int>(-1, -1);
        radius = -1.0;
    } else{
        computeCenter();
        computeRadius();
    }
}

/*! \brief Generate a string that describe the victim.
    \returns The generated string.
*/
string Victim::toString(){
    stringstream ss;
    ss << "Victim number " << value << ":\n";
    return(ss.str() + Object::toString());
}

/*! \brief Print the describing string of the victim.
*/
void Victim::print(){
    cout << toString();
}

