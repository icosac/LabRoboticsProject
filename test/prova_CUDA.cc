#include<iostream>
#include<cmath>
#include<maths.hh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <dubins_CU.hh>

using namespace std;

double* LSL (double th0, double th1, double _kmax)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);
	double tan2=atan2(C, S);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;
	double sc_s1=Angle(tan2-th0, Angle::RAD).get()*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=Angle(th1-tan2, Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;
	// return Tuple<double>(3, sc_s1.get(), sc_s2, sc_s3.get());
}

double* RSR (double th0, double th1, double _kmax)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);
	
	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;
	double sc_s1=Angle(th0-atan2(C,S), Angle::RAD).get()*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=Angle(atan2(C,S)-th1, Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;

	// return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
}

double* LSR (double th0, double th1, double _kmax)
{    
	double C = cos(th0)+cos(th1);
	double S=2*_kmax+sin(th0)+sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0, Angle::RAD).get()*invK;
	double sc_s3= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1, Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;
	// return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

double* RSL (double th0, double th1, double _kmax)
{
	double C = cos(th0)+cos(th1);
	double S=2*_kmax-sin(th0)-sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= Angle(th0-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
	double sc_s3= Angle(th1-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;
	// return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

double* RLR (double th0, double th1, double _kmax)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-Epsi>1.0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;
	double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
	double sc_s1 = Angle(th0-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
	double sc_s3 = Angle(th0-th1+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;
	// return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

double* LRL (double th0, double th1, double _kmax)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-Epsi>1.0){
	  // return Tuple<double> (0);
	  return nullptr;
	}

	double invK=1/_kmax;
	double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
	double sc_s1 = Angle(atan2(C, S)-th0+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
	double sc_s3 = Angle(th1-th0+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;

	double* ret=new double[3];
	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	return ret;

// return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

void shortest(double sc_th0, double sc_th1, double sc_Kmax, int& pidx, double* sc_s, double& Length){
	Length=DInf;
	
	vector<double* > res;
	res.push_back(RSR(sc_th0, sc_th1, sc_Kmax));
	res.push_back(LSR(sc_th0, sc_th1, sc_Kmax));
	res.push_back(RSL(sc_th0, sc_th1, sc_Kmax));
	res.push_back(RLR(sc_th0, sc_th1, sc_Kmax));
	res.push_back(LRL(sc_th0, sc_th1, sc_Kmax));
	res.push_back(LSL(sc_th0, sc_th1, sc_Kmax));
	
	int i=0; 
    for (auto value : res){
      if (value!=nullptr){
        double appL=value[0]+value[1]+value[2];
	    // cout << "appL: " << appL << endl;
        if (appL<Length){
          Length = appL;
          sc_s[0]=value[0];
          sc_s[1]=value[1];
          sc_s[2]=value[2];
          pidx=i;
        }
      }
      i++;
    }
    // COUT(Length)
}

#define SIZE 10000000000
#define THREAD 256
#define BASE 18
#define DIM (int)(log(SIZE)/log(BASE)+1)

int* toBase (int val){
	int* ret=(int*) malloc(sizeof(int)*DIM);
	int i=0;
	while(val>0){
		ret[i]=val%BASE;
		val=(int)(val/BASE);
		i++;
	}
	return ret;
}

#define CHOICE 2

#if CHOICE==1
int main (){
	cudaFree(0);
	int i=0;
	double host_time=0.0;
	double dev_time=0.0;
	for (double th0=0.0; th0<2*M_PI; th0+=0.1){
		for (double th1=0.0; th1<2*M_PI; th1+=0.1){
			for (double kmax=0.0; kmax<5; kmax+=0.1){
				// COUT(i)
				int pidx_h=-1, pidx_d=-1;
				double* dev_sc_s=(double*)malloc(sizeof(double)*3);
				double* host_sc_s=(double*)malloc(sizeof(double)*3);

				double dev_Lenght=-1;
				double host_Lenght=-1;

				auto start=Clock::now();
				shortest(th0, th1, kmax, pidx_h, host_sc_s, host_Lenght);
				auto stop=Clock::now();
				host_time+=CHRONO::getElapsed(start, stop);

				start=Clock::now();
				shortest_cuda(th0, th1, kmax, pidx_d, dev_sc_s, dev_Lenght);
				stop=Clock::now();
				dev_time+=CHRONO::getElapsed(start, stop);
				
				if (pidx_d!=pidx_h){
					cout << "th0: " << th0 << ", th1: " << th1 << ", kmax: " << kmax << ", pidx_h: " << pidx_h << ", pidx_d: " << pidx_d << endl;
					cout << "Host  Length: " << host_Lenght << " sc_s1: " << host_sc_s[0] << " sc_s2: " << host_sc_s[1] << " sc_s3: " << host_sc_s[2] << endl;
					cout << "Dev  Length: " << dev_Lenght << " sc_s1: " << dev_sc_s[0] << " sc_s2: " << dev_sc_s[1] << " sc_s3: " << dev_sc_s[2] << endl << endl;
				}	
				#define DEL 
				// if (i==DEL) return 0;
				i++;	
			}
		}
	}
	COUT(host_time)
	COUT(dev_time)
	return 0;
}


#elif CHOICE==2

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

int main (){
	Tuple<Point2<double> > points;
	points.add(Point2<double> (1.0, 0.0));
	points.add(Point2<double> (1.0, 1.0));
	points.add(Point2<double> (0.0, 1.0));
	
	Configuration2<double> start(0.0, 0.0, Angle(0.0, Angle::RAD));
	Configuration2<double> stop(0.5, 0.0, Angle(-M_PI/2.0, Angle::RAD));

	dubinsSetBest(start, stop, points, 1, 3, 6);

	// DubinsSet<double> s(start, stop, points, 1.0);

	Dubins<double> d1(start, Configuration2<double>(points.get(0), Angle(M_PI/2.0, Angle::RAD)), 1);
	Dubins<double> d2(Configuration2<double>(points.get(0), Angle(0.5*M_PI, Angle::RAD)), Configuration2<double>(points.get(1), Angle(M_PI, Angle::RAD)), 1);
	Dubins<double> d3(Configuration2<double>(points.get(1), Angle(M_PI, Angle::RAD)), Configuration2<double>(points.get(2), Angle(5.17604, Angle::RAD)), 1);
	Dubins<double> d4(Configuration2<double>(points.get(2), Angle(5.17604, Angle::RAD)), stop, 1);

	cout << d1.getId() << endl;
	cout << d1 << endl;
	cout << endl << d2.getId() << endl;
	cout << d2 << endl;
	cout << endl << d3.getId() << endl;
	cout << d3 << endl;
	cout << endl << d4.getId() << endl;
	cout << d4 << endl;

	cout << d1.length()+d2.length()+d3.length()+d4.length() << endl;
	return 0;
}
#endif