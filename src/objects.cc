#include "objects.hh"

/*float distance(Point p1, Point p2){
    return(pow( pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) ,0.5));
}*/

//_______________________________   Object   _______________________________
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

unsigned Object::size(){
    return(points.size());
}
unsigned Object::nPoint(){
    return(points.size());
}

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
    }
}
void Object::computeRadius(){
    if(points.size()>=1){
        float dist, maxRadius = center.distance(points[0]);
        for(unsigned i=1; i<points.size(); i++){
            dist = center.distance(points[i]);
            if(dist>radius){
                radius = dist;
            }
        }
        radius = maxRadius;
    }
}

void Object::offsetting(int offset){
    //documentation: http://www.angusj.com/delphi/clipper.php
    ClipperLib::Path srcPoly; //A Path represents a polyline or a polygon
    ClipperLib::Paths solution;
    for(unsigned i=0; i<points.size(); i++){
        srcPoly << ClipperLib::IntPoint(points[i].x(), points[i].y()); //Add the list of points to the Path object using operator <<
    }
    ClipperLib::ClipperOffset co;
    co.AddPath(srcPoly, ClipperLib::jtSquare, ClipperLib::etClosedPolygon); //A ClipperOffset object provides the methods to offset a given (open or closed) path

    co.Execute(solution, offset);
    //DrawPolygons(solution, 0x4000FF00, 0xFF009900);

    // save back the new polygon
    points.resize(0);
    for(unsigned i=0; i<solution.size(); i++){
        if( Orientation(solution[i]) ){ //returning true for outer polygons and false for hole polygons
            for(unsigned j=0; j<solution[i].size(); j++){
                points.push_back( Point2<int>(solution[i][j].X, solution[i][j].Y) );
            }
        }
    }
    computeCenter();
    computeRadius();
}
bool Object::insidePolyApprox(Point2<int> pt){
    //cout << "distance: " << distance(p, center) << endl;
    return((pt.distance(center)<=radius) ? true : false);
}
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
Obstacle::Obstacle(vector<Point2<int> > vp){
    points = vp;
    if(vp.size()==0){
        center = Point2<int>(-1, -1);
        radius = -1.0;
    } else{
        computeCenter();
        computeRadius();
    }
}
string Obstacle::toString(){
    return("Obstacle:\n" + Object::toString());
}
void Obstacle::print(){
    cout << toString();
}

//_______________________________   Victim   _______________________________
Victim::Victim(vector<Point2<int> > vp, int _value){
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

string Victim::toString(){
    stringstream ss;
    ss << "Victim number " << value << ":\n";
    return(ss.str() + Object::toString());
}
void Victim::print(){
    cout << toString();
}

int main(){
    /*vector<Point2<int> > o;
    o.push_back(Point2<int>(1, 1));
    o.push_back(Point2<int>(4, 3));
    Obstacle obj(o);
    obj.print();
    cout << obj.insidePolyApprox(Point2<int>(4, 1)) << endl << endl;*/


    vector<Point2<int> > v;
    v.push_back(Point2<int>(1, 1));
    v.push_back(Point2<int>(4, 2));
    v.push_back(Point2<int>(2, 7));
    Victim vict(v, 3);
    vict.print();
    Point2<int> pt(4, 4);
    cout << "Point: " << pt << "\n\tapprox: " << vict.insidePolyApprox(pt) << " - real: " << vict.insidePoly(pt) << endl;

    cout << "\nCompute offsetting\n";
    vict.offsetting(2);
    vict.print();
    cout << "Point: " << pt << "\n\tapprox: " << vict.insidePolyApprox(pt) << " - real: " << vict.insidePoly(pt) << endl;

return(0);   
}
