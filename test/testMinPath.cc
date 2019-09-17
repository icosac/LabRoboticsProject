#include <map.hh>
#include <iostream>
#include <vector>

using namespace std;

int main(){
    vector<Point2<int> > vp;
    vp.push_back(Point2<int>(  ));
    vp.push_back(Point2<int>(  ));
    vp.push_back(Point2<int>(  ));
    vp.push_back(Point2<int>(  ));
    vp.push_back(Point2<int>(  ));

    vector< vector<Point2<int> > > vvp = minPathNPointsWithChoice(vp);
return(0);   
}