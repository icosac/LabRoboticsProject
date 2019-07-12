#include <map.hh>

/*
bool checkPoint(Point2<int> p);
bool checkSegment(Point2<int> p1, Point2<int> p2);
*/

Map::Map(int _lengthX, int _lengthY, int _xx, int _yy, vector< vector<Point2<int> > > vvp){
    lengthX = _lengthX;
    lengthY = _lengthY;
    xx = _xx;
    yy = _yy;
    
    dimX = int(ceil(lengthX*1.0 / xx));
    dimY = int(ceil(lengthY*1.0 / yy));
    map = new myBit*[dimX];
    for(int i=0; i<dimX; i++){
        map[i] = new myBit[dimY];
    }

    printDimensions();
    printMap();

    for(int i=0; i<vvp.size(); i++){
        addObject(vvp[i]);
    }
    printMap();
}

void Map::addObject(vector<Point2<int> > vp){
    for(int i = 0; i<vp.size(); i++) cout << vp[i] << endl;
}

void Map::printDimensions(){
    cout << "lengthX: " << lengthX << ", lengthY: " << lengthY << endl;
    cout << "dimX: " << dimX << ", dimY: " << dimY << endl;
    cout << "xx: " << xx << ", yy: " << yy << endl;
}

string Map::matrixToString(){
    stringstream ss;
    
    for(int i=0; i<dimX; i++){
        for(int j=0; j<dimY; j++){
            ss << map[i][j] << " ";
        }
        ss << endl;
    }
    ss << endl;

    string s = ss.str();
    return(s);
}

void Map::printMap(){
    printDimensions();
    cout << matrixToString();
}