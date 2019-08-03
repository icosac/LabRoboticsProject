using namespace std;

#include <iostream>
#include <detection.hh>

Settings *sett =new Settings();

int main(){
    Mat transf;
    computeConversionParameters(transf);

    cout << "In the end:\n";// << transf << endl << endl;

return(0);
}
