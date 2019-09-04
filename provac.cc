#include<iostream>
#include<vector>
using namespace std;

double* func(){
	double* v=new double [3];
	v[0]=0;
	v[1]=1;
	v[2]=2;
	return v;
}

int main(){
	vector<double*> v;
	v.push_back(func());
	for (int i=0; i<3; i++){
		cout << v[0][i] << endl;
	}
	return 0;
}