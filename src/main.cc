// the core of the project
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "utils.hh"
#include "detection.hh"
#include "unwrapping.hh"
#include "calibration.hh"

int main (){
	calibration();
	unwrapping();
	detection();

	return 0;
}
