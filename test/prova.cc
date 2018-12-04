#define DEBUG
#include "../src/utils.hh"
#include "../src/dubins.hh"

int main (){
	CLEARFILE("data/test/prova1.test");
	CLEARFILE("data/test/prova.test");
	TOFILE("data/test/prova1.test", "ciao");
	TOFILE("data/test/prova.test", "ciao");
	return 0;
}