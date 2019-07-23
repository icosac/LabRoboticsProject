#include <map.hh>
#include <iostream>
#include <maths.hh>

using namespace std;

int main(){
    cout << "MAIN MAP\n";
    Mapp* map = new Mapp(100, 150, 5, 5);

    vector<Point2<int> > vp;
    vp.push_back(Point2<int>(13, 7));
    vp.push_back(Point2<int>(32, 7));
    //vp.push_back(Point2<int>(30, 20));
    vp.push_back(Point2<int>(14,18));

    map->addObject(vp, OBST);
    map->printMap();
    
    Point2<int> p = Point2<int>(26, 16);
    cout << "Checked point: " << p << " -> " << map->getPointType(p) << endl;

    cout << "Checked segment 1: " << map->checkSegment(p, Point2<int>(7, 7)) << endl;
    cout << "Checked segment 2: " << map->checkSegment(p, Point2<int>(18, 33)) << endl;
return(0);
}   