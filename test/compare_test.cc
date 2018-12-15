#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

void printV(vector<double> v){
	for (int i=0; i<v.size(); i++){
		printf("%f", v[i]);
		if (i!=v.size()-1){
			printf(", ");
		}
	}
	printf("\n");
}

bool compare (vector<double> v1, vector<double> v2){
	bool ret=false;
	for (int i=0; i<v1.size(); i++){
		if (v1[i]!=v2[i]){
			printf("%f!=%f\n", v1[i], v2[i]);
			ret=true;
		}
	}
	return ret;
}

vector<double> decompose(string A){
	vector<double> v;

	string delimiter = ": ";
	size_t pos = 0;

	if ((pos=A.find(delimiter))!=string::npos){
		A.erase(0,pos+delimiter.length());
	}

	delimiter=", ";

	string token;
	while ((pos = A.find(delimiter)) != string::npos) {
	    token = A.substr(0, pos);
	    v.push_back(stod(token));
	    A.erase(0, pos + delimiter.length());
	}
	return v;
}

int main (int argc, char* argv[]){
	string arg1 = argv[1];
	string arg2 = argv[2];

	ifstream f1; f1.open(arg1.c_str(), std::ifstream::in);
	ifstream f2; f2.open(arg2.c_str(), std::ifstream::in);

	// ofstream f3; f3.open("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/compare.test", ofstream::out);

	if (!f1.is_open() || !f2.is_open())
		return 1;

	string A1, A2;
	unsigned int a=1;
	while (getline(f1, A1) && getline(f2, A2)){
		if (A1.compare(A2)!=0){
			// f3 << A1 << endl << A2 << endl;
			vector <double> V1 = decompose(A1);
			vector <double> V2 = decompose(A2);

			if (compare(V1, V2)){
				cout << a << endl;
			}

			// cout << A1 << endl << A2 << endl << endl;
		}
		a++;
	}

	// f3.close();

	f1.close();
	f2.close();

	return 0;
}