#include "../src/dubins.hh"

int main (){
	Dubins<double> d=Dubins<double>();
	cout << d << endl;
	d.shortest_path();
	return 0;
}