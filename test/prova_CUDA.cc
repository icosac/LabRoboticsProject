#include<iostream>
#include<cmath>
#include<maths.hh>

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

	if (equal(fabs(temp1), 1.0) ){
	  temp1=round(temp1);
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

	if (equal(fabs(temp1), 1.0) ){
	  temp1=round(temp1);
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

void shortest(double sc_th0, double sc_th1, double sc_Kmax, int& pidx){
	double Length=DInf;
	double sc_s1=0.0;
	double sc_s2=0.0;
	double sc_s3=0.0;
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
        if (appL<Length){
          Length = appL;
          sc_s1=value[0];
          sc_s2=value[1];
          sc_s3=value[2];
          pidx=i;
        }
      }
      i++;
    }
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

int main (){
	for (double th0=0.0; th0<2*M_PI; th0+=0.1){
		for (double th1=0.0; th1<2*M_PI; th1+=0.1){
			for (double kmax=0.0; kmax<5; kmax+=0.1){
				int pidx=-1;
				cout << "th0: " << th0 << ", th1: " << th1 << ", kmax: " << kmax << endl;
				shortest(th0, th1, kmax, pidx);
				
			}
		}
	}
	return 0;
}