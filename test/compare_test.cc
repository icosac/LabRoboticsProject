#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main (int argc, char* argv[]){
	string arg1 = argv[1];
	string arg2 = argv[2];

	ifstream f1; f1.open(arg1.c_str(), std::ifstream::in);
	ifstream f2; f2.open(arg2.c_str(), std::ifstream::in);

	ofstream f3; f3.open("/Users/enrico/Desktop/prova.txt", ofstream::out);

	if (!f1.is_open() || !f2.is_open())
		return 1;

	string A1, A2;
	unsigned int a=0;
	while (getline(f1, A1) && getline(f2, A2) && a<80100){
		cout << a << endl;
		f3 << A1 << endl;
		if (A2.find("100.000")!=string::npos || A2.find("150.000")!=string::npos) {
			while(true) {
				getline(f2, A2);
				if (A2.find("100.000")==string::npos && A2.find("150.000")==string::npos) {
					break;
				}
			}
		}
		if (A1.compare(A2)!=0){
			cout << A1 << endl << A2 << endl << endl;
			cout << "A: " << a << endl;
			return 1;
		}
		a++;
	}

	f3.close();

	f1.close();
	f2.close();

	return 0;
}