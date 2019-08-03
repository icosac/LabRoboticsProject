using namespace std;

#include <iostream>
#include <detection.hh>

Settings *sett =new Settings();

int main(){
    Point2<int> p;
    for(int i=0; i<5; i++){
        p = localize();
    }

return(0);
}
