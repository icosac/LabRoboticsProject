#include<iostream>

using namespace std;

int main (){
	for (int i=2; i<=134217728; i*=2){
		string cmd="./vector_sum.out "+to_string(i); 
		system(cmd.c_str());
	}
	return 0;
}