#include <iostream>
#include <cmath>
#include "maths.hh"

using namespace std;

int main(int argc, const char * argv[]) {
	cout << "ANGLES" << endl;
	Angle a = Angle ();
	cout << "Empty angle: " << a << endl;
	Angle b (100, Angle::DEG);
	cout << "DEG angle: " << b << endl;
	Angle c (M_PI, Angle::RAD);
	cout << "RAD angle: " << c << endl;
	cout << "Dim DEG angle: " << b.get() << endl;
	cout << "Dim RAD angle: " << c.get() << endl;
	cout << "Type DEG angle: " << b.getType() << "==" << b.getTypeName() << endl;
	cout << "Type RAD angle: " << c.getType() << "==" << c.getTypeName() << endl;
	double th_DEG=200;
	b.set(th_DEG); cout << "Set new angle to " << th_DEG << "==" << b.get() << endl;
	double th_RAD=2*M_PI;
	c.set(th_RAD); cout << "Set new angle to " << th_RAD<< "==" << c.get() << endl;
	th_DEG=400;
	b.set(th_DEG); cout << "Set new angle to " << th_DEG << "==" << b.get() << endl;
	th_RAD=2*M_PI+1;
	c.set(th_RAD); cout << "Set new angle to " << th_RAD<< "==" << c.get() << endl;
	
	cout << b << " "; b.radToDeg(); cout << "Nothing should happen " << b << endl;
	cout << c << " "; c.degToRad(); cout << "Nothing should happen " << c << endl;
	cout << b << " "; b.degToRad(); cout << "It should be in RAD now " << b << endl;
	cout << c << " "; c.radToDeg(); cout << "It should be in DEG now " << c << endl;

	cout << "The value in DEG: " << b.toDeg() << endl;
	cout << "The value in RAD: " << c.toRad() << endl;

	b.set(M_PI);
	c.set(90);

	cout << "b: " << b << endl;
	cout << "c: " << c << endl;

	cout << "cos(" << b << ")=" << b.cos() << endl; 
	cout << "cos(" << c << ")=" << c.cos() << endl; 
	cout << "sin(" << b << ")=" << b.sin() << endl; 
	cout << "sin(" << c << ")=" << c.sin() << endl; 
	cout << "tan(" << b << ")=" << b.tan() << endl; 
	cout << "tan(" << c << ")=" << c.tan() << endl; 

	b+=c; cout << "b+=c: " << b << endl;
	b-=c; cout << "b-=c: " << b << endl;
	c+=b; cout << "c+=b: " << c << endl;
	c-=b; cout << "c-=b: " << c << endl;

	b/=2; cout << "b/=2: " << b << endl;
	c/=2; cout << "c/=2: " << c << endl;
	b*=2; cout << "b*=2: " << b << endl;
	c*=2; cout << "c*=2: " << c << endl;

	b/=0.5; cout << "b/=0.5: " << b << endl;
	c/=0.5; cout << "c/=0.5: " << c << endl;
	b*=0.5; cout << "b*=0.5: " << b << endl;
	c*=0.5; cout << "c*=0.5: " << c << endl;
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////

	cout << endl << endl << endl << "TUPLES" << endl; 
	Tuple<int> d = Tuple<int> ();
	cout << "Empty Tuple: " << d << endl;
	Tuple<int> e (2, 1, 2);
	cout << "int Tuple: " << e << endl;
	Tuple<double> f (3, 1.1, 2.2, 3.3);
	cout << "double Tuple: " << f << endl;
	cout << "Dim int Tuple: " << e.size() << endl;
	cout << "Dim double Tuple: " << f.size() << endl;
	cout << "Get out element" << f.get(f.size()) << endl;
	cout << "Get right elements " << f.get(0) << "  " << e.get(1) << endl;

	f.set(1, 4.4); cout << f << endl;
	e.add(3.3); cout << e << endl;
	e.remove(2); cout << e << endl;
	e.add(3); 

	cout << "e distance:" << endl;
	cout << e.distance(f) << endl;
	cout << e.distance(f, DISTANCE_TYPE::EUCLIDEAN) << endl;
	cout << e.distance(f, DISTANCE_TYPE::MANHATTAN) << endl;
	cout << e.EuDistance(f) << endl;
	cout << e.MaDistance(f) << endl;

	cout << "f distance:" << endl;
	cout << f.distance(e) << endl;
	cout << f.distance(e, DISTANCE_TYPE::EUCLIDEAN) << endl;
	cout << f.distance(e, DISTANCE_TYPE::MANHATTAN) << endl;
	cout << f.EuDistance(e) << endl;
	cout << f.MaDistance(e) << endl;

	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////

	cout << endl << endl << endl << "POINT2" << endl;
	Point2<int> g = Point2<int> ();
	cout << "Empty point: " << g << endl;
	Point2<int> h (1, 2);
	cout << "int point: " << h << endl;
	Point2<double> i (2.2, 3.3);
	cout << "double point: " << i << endl; 
	h.x(2); h.y(1);
	cout << "h: " << h << endl;
	i.x(3.3); i.y(2.2);
	cout << "i: " << i << endl;
	cout << h.offset((2/sqrt(2)), Angle(45, Angle::DEG)) << "  3, 2==" << h << endl;
	cout << i.offset((2/sqrt(2)), Angle(45, Angle::DEG)) << "  4.3, 3.2==" << i << endl; 

	cout << h.offset(Point2<int>(-1, -1)) << "  2, 1==" << h << endl;
	cout << h.offset(Point2<int>(-1.1, -1.1)) << "  1, 0==" << h << endl;
	cout << i.offset(Point2<double>(-1.1, -1.1)) << "  3.2, 2.1==" << i << endl; 

	cout << h.offset_x(1) << h.offset_y(2.1) << "  2, 2==" << h << endl;
	cout << i.offset_x(1) << i.offset_y(2.1) << "  4.2, 4.2==" << i << endl;

	cout << h.distance(i) << endl;
	cout << i.distance(h) << endl;

	cout << h.distance(i, DISTANCE_TYPE::MANHATTAN) << endl;
	cout << i.distance(h, DISTANCE_TYPE::MANHATTAN) << endl;	

  return 0;
}
