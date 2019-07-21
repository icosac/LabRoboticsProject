// the core of the project
#include <detection.hh>
#include <unwrapping.hh>
#include <configure.hh>

#include<iostream>
using namespace std;

int main (){
  cout <<"Configure" << endl;
  configure();
	cout << "unwrapping" << endl;
	unwrapping();
	cout << "detection" << endl;
	detection();
	cout << "end" << endl;

	return 0;
}
