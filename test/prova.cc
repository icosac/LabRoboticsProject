#include <configure.hh>

int main(){
  Settings *s=new Settings();
  cout << *s << endl;
  s->readFromFile();
  cout << *s << endl;

  return 0;
}