// the core of the project
#include <detection.hh>
#include <unwrapping.hh>
#include <configure.hh>

#include<iostream>
using namespace std;

//TODO create global settings

int main (){
  cout <<"Configure" << endl;
  configure(true);
	cout << "unwrapping" << endl;
	unwrapping();
	cout << "detection" << endl;
	detection();
	cout << "end" << endl;

	return 0;
}
