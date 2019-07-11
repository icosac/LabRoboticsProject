// the core of the project
#include <detection.hh>
#include <unwrapping.hh>
#include <calibration.hh>

int main (){
	calibration();
	unwrapping();
	detection();

	return 0;
}
