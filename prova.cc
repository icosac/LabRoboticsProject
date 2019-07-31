#include<iostream>

using namespace std;

class prova
{
private:
public:
	int* a;
	prova() { a=new int [2]; };
	~prova() { delete [] a; };

};

int main (){
	prova c=prova();
	cout << sizeof(c.a) << endl;
	cout << sizeof(int*) << endl;	
	return 0;
}