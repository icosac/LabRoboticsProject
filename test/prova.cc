#include <iostream>
#include <maths.hh>

using namespace std;

void disp (Tuple<int>& z, int i, int N){
  if (i==0){
    for (uint a=1; a<N+1; a++){
      t.addIfNot(z);
      z[i]=a;
      if (a==N){
        t.addIfNot(z);
      }
    }
    z[i]=0;
  }
  else {
    for (uint a=1; a<N+1; a++){
      disp(z, i-1, N);
      z[i]=a;
      if (a==N)
        disp(z, i-1, N);
    }
    z[i]=0;
  }
}

int main(){
  int size=5;
  
  Tuple<int> z=Tuple<int>(size, 0, 0, 0, 0);
	#define N 4
  uint n_guess=pow(size, N);

  disp(z, size-1, N);
  cout << endl << endl << endl << endl;
  for (auto el : t){
    cout << el << endl;
  }
  cout << "size: " << t.size() << endl;
  return 0;
}