#include <dubins_CU.hh>

__device__ double mod2pi (double angle){
	while(angle>=2*M_PI){
		angle-=(M_PI*2);
	}
	while(angle<0){
		angle+=(M_PI*2);
	}
	ret angle;
}

__global__ void LSL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);
	double tan2=atan2(C, S);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret=nullptr;
	}

	double invK=1/_kmax;
	double sc_s1=mod2pi(tan2-th0)*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=mod2pi(th1-tan2)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

}

__global__ void RSR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret=nullptr;
	}

	double invK=1/_kmax;
	double sc_s1=mod2pi(th0-atan2(C,S))*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=mod2pi(atan2(C,S)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;


}

__global__ void LSR (double th0, double th1, double _kmax, double* ret)
{    
	double C = cos(th0)+cos(th1);
	double S=2*_kmax+sin(th0)+sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  ret=nullptr;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= mod2pi(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0)*invK;
	double sc_s3= mod2pi(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

}

__global__ void RSL (double th0, double th1, double _kmax, double* ret)
{
	double C = cos(th0)+cos(th1);
	double S=2*_kmax-sin(th0)-sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  ret=nullptr;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= mod2pi(th0-atan2(C,S)+atan2(2, _kmax*sc_s2))*invK;
	double sc_s3= mod2pi(th1-atan2(C,S)+atan2(2, _kmax*sc_s2))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

}

__global__ void RLR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-Epsi>1.0){
	  ret=nullptr;
	}

	if (equal(fabs(temp1), 1.0) ){
	  temp1=round(temp1);
	}

	double invK=1/_kmax;
	double sc_s2 = mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = mod2pi(th0-atan2(C, S)+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = mod2pi(th0-th1+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

}

__global__ void LRL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-Epsi>1.0){
	  ret=nullptr;
	}

	if (equal(fabs(temp1), 1.0) ){
	  temp1=round(temp1);
	}

	double invK=1/_kmax;
	double sc_s2 = mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = mod2pi(atan2(C, S)-th0+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = mod2pi(th1-th0+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;


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