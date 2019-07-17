#include <map.hh>
#include <iostream>
#include <bitset>         // std::bitset

using namespace std;

int main(){
    cout << "MAIN MAP\n";
    Mapp* map = new Mapp(150, 100, 5, 5);

    vector<Point2<int> > vp;
    vp.push_back(Point2<int>(13, 7));
    vp.push_back(Point2<int>(32, 7));
    //vp.push_back(Point2<int>(30, 20));
    vp.push_back(Point2<int>(14,18));

    map->addObject(vp, OBST);
    map->printMap();
    
    Point2<int> p = Point2<int>(26, 16);
    cout << "Checked point: " << p << " -> " << map->getPointType(p) << endl;
return(0);
}   