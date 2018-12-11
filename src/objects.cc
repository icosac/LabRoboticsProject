#include "objects.hh"

float distance(Point p1, Point p2){
    return(pow( pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) ,0.5));
}

//_______________________________   Object   _______________________________
string Object::toString(){
    stringstream ss;
    ss << "objects:\ncenter: [" << center.x << ", " << center.y << "], radius=" << radius << "\n";
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
void Object::print(){
    cout << toString();
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
    return("Obstacle " + Object::toString());
}
void Obstacle::print(){
    cout << toString();
}

int main(){
    vector<Point> v;
    v.push_back(Point(1, 1));
    v.push_back(Point(4, 3));
    Obstacle obj(v);
    obj.print();
    cout << obj.collideApproximate(Point(6, 1)) << endl;
return(0);   
}/*/
