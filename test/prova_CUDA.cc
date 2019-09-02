#include<iostream>
#include<cmath>
#include<fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// #include<maths.hh>
#include <dubins.hh>
#include <dubins_CU.hh>

using namespace std;

double elapsedScale=0;
double elapsedPrimitives=0;
double elapsedBest=0;
double elapsedArcs=0;
double elapsedCheck=0;
unsigned long countTries=0;
double elapsedTupleSet=0.0;
double elapsedTuple=0.0;
double elapsedVar=0;
double elapsedCirc=0;
double elapsedSet=0;
double elapsedLSL=0;
double elapsedRSR=0;
double elapsedLSR=0;
double elapsedRSL=0;
double elapsedRLR=0;
double elapsedLRL=0;

#define SCALE 1.0
#define ES 2

typedef double TYPE;

int main (){
	cudaFree(0);
	Tuple<Point2<TYPE> > points;
	Configuration2<TYPE> start;
	Configuration2<TYPE> end;

	#if ES==1
	start=Configuration2<TYPE>(0*SCALE,0*SCALE, Angle(-M_PI/3.0, Angle::RAD));
	points.add(Configuration2<TYPE>(-0.1*SCALE,0.3*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.2*SCALE,0.8*SCALE, Angle()));
	end=Configuration2<TYPE>(1.0*SCALE,1.0*SCALE, Angle(-M_PI/6.0, Angle::RAD));
	double kmax=3/SCALE;  
	#define ESRES 3.415578858075

	#elif ES==2 
	start=Configuration2<TYPE>(0*SCALE,0*SCALE, Angle(-M_PI/3.0, Angle::RAD));
	points.add(Configuration2<TYPE>(-0.1*SCALE,0.3*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.2*SCALE,0.8*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE,1.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.5*SCALE,0.5*SCALE, Angle()));
	end=Configuration2<TYPE>(0.5*SCALE,0.0*SCALE, Angle(-M_PI/6.0, Angle::RAD));
	double kmax=3/SCALE;
	#define ESRES 6.278034550309

	#elif ES==3
	start=Configuration2<TYPE>(0.5*SCALE, 1.2*SCALE, Angle(5.0*M_PI/6.0, Angle::RAD));
	points.add(Configuration2<TYPE>(0.0*SCALE, 0.8*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.0*SCALE, 0.4*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.1*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.4*SCALE, 0.2*SCALE, Angle()));

	points.add(Configuration2<TYPE>(0.5*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.6*SCALE, 1.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE, 0.8*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.4*SCALE, 0.2*SCALE, Angle()));

	points.add(Configuration2<TYPE>(1.2*SCALE, 1.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.5*SCALE, 1.2*SCALE, Angle()));
	points.add(Configuration2<TYPE>(2.0*SCALE, 1.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.5*SCALE, 0.8*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.5*SCALE, 0.0*SCALE, Angle()));

	points.add(Configuration2<TYPE>(1.7*SCALE, 0.6*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.9*SCALE, 1.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(2.0*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.9*SCALE, 0.0*SCALE, Angle()));
	end=Configuration2<TYPE>(2.5*SCALE, 0.6*SCALE, Angle());
	double kmax=3/SCALE;
	#define ESRES 11.916212654286

	#elif ES==4
	start=Configuration2<TYPE>(0.5*SCALE, 1.2*SCALE, Angle(5.0*M_PI/6.0, Angle::RAD));
	points.add(Configuration2<TYPE>(0.0*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.5*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.5*SCALE, 0.5*SCALE, Angle()));
	points.add(Configuration2<TYPE>(2.0*SCALE, 0.5*SCALE, Angle()));
	
	points.add(Configuration2<TYPE>(2.0*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.5*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(1.0*SCALE, 0.0*SCALE, Angle()));
	points.add(Configuration2<TYPE>(0.0*SCALE, 0.0*SCALE, Angle()));
	end=Configuration2<TYPE>(0.0*SCALE, 0.5*SCALE, Angle());
	double kmax=3/SCALE;
	#define ESRES 7.467562181965
	#endif

	auto start_t=Clock::now();
	double* anglss=dubinsSetCuda(start, end, points, kmax, 1, points.size(), 360);
	auto stop_t=Clock::now();
	double elapsedCuda=CHRONO::getElapsed(start_t, stop_t);

	cout << "elapsedCuda: " << elapsedCuda << endl; 

	Tuple<Configuration2<double> > confs;
	cout << start.angle().toRad() << " ";
	confs.add(start);
	for (int i=1; i<points.size()+1; i++){
		cout << anglss[i] << " ";
		confs.add(Configuration2<double> (points.get(i-1), Angle(anglss[i], Angle::RAD)));
	}
	cout << end.angle().toRad() << endl;
	confs.add(end);

	DubinsSet<double> s_CUDA (confs, kmax);
	COUT(s_CUDA)
	cout << "Length: " << s_CUDA.getLength() << endl; 
	cout << "Error: " << s_CUDA.getLength()-ESRES << endl;
	COUT(s_CUDA.getKmax())

	uint DIMY=750;
	uint DIMX=500;
	Mat map(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
	s_CUDA.draw(DIMX, DIMY, map, 250, 2.5);
	my_imshow("CUDA", map, true);
	mywaitkey();

	// dubinsSetBest(start, end, points, 1, 4, 8, kmax);
	// DubinsSet<double> s(start, end, points, 8, kmax);
	return 0;

	// for (double i=2.0; i<=16.0; i*=2.0){
	// 	if (i==512.0){
	// 		i=360.0;
	// 	}
	// 	ofstream out_data; out_data.open("data/test/CUDA.test", fstream::app);
	// 	out_data << endl << endl;
	// 	out_data << "Parts: " << i << endl;
	// 	cout << "Parts: " << i << endl;
	// 	double* angles=(double*) malloc(sizeof(double)*(points.size()+2));

	// 	auto start_t=Clock::now();
	// 	// angles=dubinsSetBest(start, end, points, 1, points.size(), i, kmax);
	// 	auto stop_t=Clock::now();
	// 	double elapsedCuda=CHRONO::getElapsed(start_t, stop_t);
		
	// 	auto _start_t=Clock::now();
	// 	// DubinsSet<double> s(start, end, points, i, kmax);
	// 	auto _stop_t=Clock::now();
	// 	double elapsedCPP=CHRONO::getElapsed(_start_t, _stop_t);

	// 	out_data << "elapsedCuda: " << elapsedCuda << endl;
	// 	out_data << "elapsedCPP: " << elapsedCPP << endl << endl;
	// 	out_data.close();

	// 	Tuple<Configuration2<double> > confs;
	// 	cout << start.angle().toRad() << " ";
	// 	confs.add(start);
	// 	for (int i=1; i<points.size()+1; i++){
	// 		cout << angles[i] << " ";
	// 		confs.add(Configuration2<double> (points.get(i-1), Angle(angles[i], Angle::RAD)));
	// 	}
	// 	cout << end.angle().toRad() << endl;
	// 	confs.add(end);

	// 	DubinsSet<double> s_CUDA (confs, kmax);

	// 	uint DIMY=750;
	// 	uint DIMX=500;
	// 	Mat map(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
	// 	s_CUDA.draw(DIMX, DIMY, map, 250, 2.5);
	// 	my_imshow("CUDA", map, true);
	// 	mywaitkey();
	// }

	return 0;
}
