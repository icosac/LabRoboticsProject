#include<iostream>
#include<cmath>
#include<fstream>

#include<maths.hh>
#include<dubins.hh>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

#define SCALE 1
#define ES 1

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

	#endif
	
// 	auto start_t=Clock::now();
// 	dubinsSetBest(start, end, points, 1, points.size(), 16, kmax);
// // 	Tuple<Configuration2<double > > app_confs;
// // 	app_confs.add(start);
// // 	app_confs.add(Configuration2<double> (-0.1, 0.3, Angle(1.03038, Angle::RAD))); 
// // 	app_confs.add(Configuration2<double> (0.2, 0.8, Angle(1.03038, Angle::RAD)));
// // 	app_confs.add(end);
// // #define ES1RES 3.415578858075

// // 	DubinsSet<double> appSet(app_confs, 3.0);
// // 	COUT(appSet)
// // 	COUT(appSet.getLength()-ES1RES)
// 	auto stop_t=Clock::now();
// 	double elapsedCuda=CHRONO::getElapsed(start_t, stop_t, CHRONO::MSEC);
// 	cout << "elapsed: " << elapsedCuda << endl;
	// DubinsSet<double> s(start, end, points, 16, kmax);


	// return 0;
	// vector<double> v={4, 16, 90};
	vector<double> v={4};
	for (auto i : v){
		ofstream out_data; out_data.open("data/test/CUDA.test", fstream::app);
		out_data << endl << endl;
		out_data << "Parts: " << i << endl;
		cout << "Parts: " << i << endl;
		double* angles=(double*) malloc(sizeof(double)*(points.size()+2));

		auto start_t=Clock::now();
		angles=dubinsSetBest(start, end, points, 1, points.size(), i, kmax);
		auto stop_t=Clock::now();
		double elapsedCuda=CHRONO::getElapsed(start_t, stop_t);
		out_data << "elapsedCuda: " << CHRONO::getElapsed(start_t, stop_t, "", CHRONO::MSEC) << endl;
		
		// auto _start_t=Clock::now();
		// DubinsSet<double> s(start, end, points, i, kmax);
		// auto _stop_t=Clock::now();
		// double elapsedCPP=CHRONO::getElapsed(_start_t, _stop_t);
		// out_data << "elapsedCPP: " << CHRONO::getElapsed(_start_t, _stop_t, "", CHRONO::MSEC) << endl << endl;
		// out_data.close();

		Tuple<Configuration2<double> > confs;
		cout << start.angle().toRad() << " ";
		confs.add(start);
		for (int i=1; i<points.size()+1; i++){
			cout << angles[i] << " ";
			confs.add(Configuration2<double> (points.get(i-1), Angle(angles[i], Angle::RAD)));
		}
		cout << end.angle().toRad() << endl;
		confs.add(end);
		cout << confs << endl;

		DubinsSet<double>* s_CUDA=new DubinsSet<double> (confs, kmax);
		cout << "Length: " << s_CUDA->getLength() << " Error: " << s_CUDA->getLength()-ESRES << endl;
		cout << *s_CUDA << endl;

		uint DIMY=750;
		uint DIMX=500;
		Mat map(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
		s_CUDA->draw(DIMX, DIMY, map, 250, 2.5);
		my_imshow("CUDA", map, true);
		mywaitkey();
		delete s_CUDA;
	}

	return 0;
}


/*
double* angles=(double*) malloc(sizeof(double)*4);
angles=dubinsSetBest(start, end, points, 1, 2, 90, kmax);

DubinsSet<double> s(start, end, points, 90, kmax);

Tuple<Configuration2<double> > confs;
cout << start.angle().toRad() << " ";
confs.add(start);
for (int i=1; i<3; i++){
	cout << angles[i] << " ";
	confs.add(Configuration2<double> (points.get(i-1), Angle(angles[i], Angle::RAD)));
}
cout << end.angle().toRad() << endl;
confs.add(end);

DubinsSet<double> s_CUDA (confs, kmax);

uint DIMY=750;
uint DIMX=500;
Mat map(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
s_CUDAdraw(DIMX, DIMY, map, 250, 2.5);
my_imshow("CUDA", map, true);
mywaitkey();
*/