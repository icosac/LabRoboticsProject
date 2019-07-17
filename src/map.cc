#include <map.hh>

/*
bool checkPoint(Point2<int> p);
bool checkSegment(Point2<int> p1, Point2<int> p2);
*/

Mapp::Mapp(int _lengthX, int _lengthY, int _pixX, int _pixY, vector< vector<Point2<int> > > vvp){
    lengthX = _lengthX;
    lengthY = _lengthY;
    pixX = _pixX;
    pixY = _pixY;
    
    dimX = int(ceil(lengthX*1.0 / pixX));
    dimY = int(ceil(lengthY*1.0 / pixY));
    map = new myBit*[dimY];
    for(int i=0; i<dimY; i++){
        map[i] = new myBit[dimX];
    }

    printDimensions();
    //printMap();

    for(uint i=0; i<vvp.size(); i++){
        addObject(vvp[i], OBST);
    }
    //printMap();
}

void Mapp::addObject(vector<Point2<int> > vp, const OBJ_TYPE type){    
    if(vp.size()>=3){
        vp.push_back(vp[0]);
        cout << "Points:\n";
        for(uint i = 0; i<vp.size(); i++){
            cout << "\t" << vp[i] << endl;
        }
        cout << endl;

        //min, max of x, y computed
        int jMin=dimX, iMin=dimY, jMax=0, iMax=0;

        // for each side of the object
        for(uint s=0; s<vp.size()-1; s++){
            cout << "\n_______________________________________________________\nPoint n:" << s+1 << endl;
            //save the end points of the segment

            // save in p0 the point with the lowest x
            Point2<int> p0, p1;
            if( vp[s].x() < vp[s+1].x() ){
                p0 = vp[s];
                p1 = vp[s+1];
            } else{
                p0 = vp[s+1];
                p1 = vp[s];
            }

            // x and y of point 0 and 1
            int x0=p0.x(), x1=p1.x();
            double y0=p0.y(), y1=p1.y();
            cout << x0 << "," << y0 << " - " << x1 << "," << y1 << endl;
            double th = atan2(y1-y0, x1-x0);

            int j = x0/pixX;
            int i = int(y0)/pixY;
            cout << "i: " << i << ", j: " << j << endl;

            //for
            while(!(x0==x1 && equal(y0, y1, 0.0001))){
                cout << "check x: " << (x0==x1) << ", check y: " << equal(y0, y1) << endl;
                cout << "check fail on -> " << x0 << "," << y0 << " - " << x1 << "," << y1 << endl;
                //localize the left point on the grid
                cout << "coloring pixel: " << i << ", " << j << endl;
                map[i][j] = type;

                // compute the y given the x
                int x = min(pixX*(j+1), x1);
                cout << "x: " << x;
                
                double l = (x-x0)*1.0/cos(th);
                double y;
                // to manage the situation of a vertical line
                if(int(l)==0){
                    y = y1;
                } else{
                    y = y0 + l*sin(th);
                }
                cout << ", y: " << y << endl;
                
                // color the whole line between the (j, i (aka i0)) and (j, i1)
                int i1 = int(floor(y))/pixY;
                cout << "i: " << i << " i1: " << i1 << endl;
                cout << "for: \n";
                for(int k=min(i, i1); k<=max(i, i1); k++){
                    cout << "\t" << k << " " << j << endl;
                    map[k][j] = type;
                }
                cout << endl;

                x0 = x;
                y0 = y;
                j++;
                i = i1;
            }

            // todo update min & max
        }
    } else{
        cout << "Invalid object less than 3 sides!\n";
    }
}

void Mapp::printDimensions(){
    cout << "lengthX: " << lengthX << ", lengthY: " << lengthY << endl;
    cout << "dimX: " << dimX << ", dimY: " << dimY << endl;
    cout << "pixX: " << pixX << ", pixY: " << pixY << endl;
}

string Mapp::matrixToString(){
    stringstream ss;
    
    for(int i=0; i<dimY; i++){
        for(int j=0; j<dimX; j++){
            ss << map[i][j] << " ";
        }
        ss << endl;
    }
    ss << endl;

    string s = ss.str();
    return(s);
}

void Mapp::printMap(){
    printDimensions();
    cout << matrixToString();
}