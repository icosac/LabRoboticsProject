#include "objects.hh"

float distance(Point p1, Point p2){
    return(pow( pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) ,0.5));
}

//_______________________________   Object   _______________________________
string Object::toString(){
    stringstream ss;
    ss << "center: [" << center.x << ", " << center.y << "], radius=" << radius << "\n";
    if(points.size()>0){
        ss << points.size() << " points:\n[";
        for(unsigned i=0; i<points.size(); i++){
            ss << "(" << points[i].x << ", " << points[i].y << "), ";
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
        int minX=points[0].x, maxX=points[0].x;
        int minY=points[0].y, maxY=points[0].y;
        
        for(unsigned i=1; i<points.size(); i++){
            if(points[i].x<minX){
                minX = points[i].x;
            }
            if(points[i].x>maxX){
                maxX = points[i].x;
            }

            if(points[i].y<minY){
                minY = points[i].y;
            }
            if(points[i].y>maxY){
                maxY = points[i].y;
            }
        }
        center.x = (int)round((minX+maxX)/2.0);
        center.y = (int)round((minY+maxY)/2.0);
    }
}
void Object::computeRadius(){
    if(points.size()>=1){
        float dist, maxRadius = distance(center, points[0]);
        for(unsigned i=1; i<points.size(); i++){
            dist = distance(center, points[i]);
            if(dist>radius){
                radius = dist;
            }
        }
        radius = maxRadius;
    }
}

void Object::offsetting(int offset){
    ClipperLib::Path srcPoly; //A Path represents a polyline or a polygon
    ClipperLib::Path solution;
    for(unsigned i=0; i<points.size(); i++){
        srcPoly << ClipperLib::IntPoint(points[i].x, points[i].y); //Add the list of points to the Path object using operator <<
    }
    /*ClipperLib::ClipperOffset co;
    co.AddPath(srcPoly, ClipperLib::jtSquare, ClipperLib::etClosedPolygon); //A ClipperOffset object provides the methods to offset a given (open or closed) path

    co.Execute(solution, -7.0);*/
}
bool Object::collideApproximate(Point p){
    //cout << "distance: " << distance(p, center) << endl;
    return((distance(p, center)<=radius) ? true : false);
}

//_______________________________   Obstacle   _______________________________
Obstacle::Obstacle(vector<Point> vp){
    points = vp;
    if(vp.size()==0){
        center = Point(-1, -1);
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
Victim::Victim(vector<Point> vp, int _value){
    points = vp;
    value = _value;
    if(vp.size()==0){
        center = Point(-1, -1);
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

//
int main(){
    // code taken from: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/_Body.htm
    ClipperLib::Path subj;
    ClipperLib::Paths solution;
    subj << 
        ClipperLib::IntPoint(348,257) << ClipperLib::IntPoint(364,148) << ClipperLib::IntPoint(362,148) << 
        ClipperLib::IntPoint(326,241) << ClipperLib::IntPoint(295,219) << ClipperLib::IntPoint(258,88) << 
        ClipperLib::IntPoint(440,129) << ClipperLib::IntPoint(370,196) << ClipperLib::IntPoint(372,275);
    ClipperLib::ClipperOffset co;
    co.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    co.Execute(solution, -7.0);
   
//   //draw solution ...
//   DrawPolygons(solution, 0x4000FF00, 0xFF009900);
}/*/
int main(){
    vector<Point> v;
    v.push_back(Point(1, 1));
    v.push_back(Point(4, 3));
    Obstacle obj(v);
    obj.print();
    cout << obj.collideApproximate(Point(6, 1)) << endl << endl;

    Victim vict(v, 3);
    vict.print();
    cout << obj.collideApproximate(Point(6, 1)) << endl;
return(0);   
}//*/
