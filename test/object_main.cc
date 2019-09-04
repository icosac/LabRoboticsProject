#include <objects.hh>
#include <iostream>

using namespace std;

//_______________________________   MAIN   _______________________________
int main(){ // basic main that test the functionalities of the classes
    vector<Point2<int> > o;
    o.push_back(Point2<int>(1, 1));
    o.push_back(Point2<int>(4, 3));

    Obstacle obj(o);
    obj.print();
    cout << obj.insidePolyApprox(Point2<int>(4, 1)) << endl << endl;


    // vector<Point2<int> > v;
    // v.push_back(Point2<int>(1, 1));
    // v.push_back(Point2<int>(4, 2));
    // v.push_back(Point2<int>(2, 7));

    // Victim vict(v, 3);
    // vict.print();

    // Point2<int> pt(4, 4);
    // cout << "Point: " << pt << "\n\tapprox: " << vict.insidePolyApprox(pt) << " - real: " << vict.insidePoly(pt) << endl;

    // cout << "\nCompute offsetting\n";
    // vict.offsetting(2);
    // vict.print();
    // cout << "Point: " << pt << "\n\tapprox: " << vict.insidePolyApprox(pt) << " - real: " << vict.insidePoly(pt) << endl;

return(0);   
}