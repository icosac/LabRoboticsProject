//#include <map.hh>
#include <iostream>
#include <bitset>         // std::bitset

using namespace std;

void test(int xx, int yy){
    int **map = new int*[xx];
    for(int i = 0; i < xx; i++) {
        map[i] = new int[yy];
    }
    map[1][2] = 5;
    for(int i=0; i<xx; i++){
        for(int j=0; j<yy; j++){
            cout << map[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(){
    cout << "MAIN MAP\n";
    test(3, 5);
    int a = 5;
    short int b = 3;
    bool c = 1;
    cout << sizeof(a) << sizeof(b) << sizeof(c) << endl;
}