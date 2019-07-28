#include<dubins.hh>

using namespace std;

int main(){
	Configuration2<double> C0(100.0, 100.0, Angle(315, Angle::DEG));
	Configuration2<double> C1(300.0, 300.0, Angle(251.565, Angle::DEG));

	Dubins<double> d(C0, C1, 0.01);

	cout << "id: " << d.getId() << endl;

	cout << d << endl;

	return 0;
}



// extern const Angle A_DEG_NULL;
// extern const Angle A_360;
// for (int x0=0; x0<1000; x0+=10){
// 	for (int y0=0; y0<1500; y0+=10){
// 		for (Angle a0=A_DEG_NULL; a0!=A_360; a0+=Angle(10, Angle::DEG)){
// 			for (int x1=0; x1<1000; x1+=10){
// 				for (int y1=0; y1<1500; y1+=10){
// 					for (Angle a1=A_DEG_NULL; a1!=A_360; a1+=Angle(10, Angle::DEG)){
// 						Dubins d=()
// 					}
// 				}
// 			}
// 		}
// 	}
// }