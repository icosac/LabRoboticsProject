#define pow2(x) x*x
#define Epsi DBL_EPSILON

double mod2pi (double phi){
	double out=phi;
  while (out<0) {
    out=out+2*M_PI;
  } 
  while (out>=2*M_PI) {
    out=out-2*M_PI;
  }
  return out;
}

__kernel void LSL (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax, 
										__global double* ret)
{
  double C=cos((double)(*th1))-cos((double)(*th0));
  double S=2*(*_kmax)+sin((double)(*th0))-sin((double)(*th1));
  double temp1=2+4*pow2(*_kmax)-2*cos((double)(*th0)-(*th1))+4*(*_kmax)*(sin((double)(*th0))-sin((double)(*th1)));

  if (temp1<0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
    } else {
    
    double invK=1/(*_kmax);
    double sc_s1=mod2pi(atan2(C,S)-(*th0))*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=mod2pi((*th1)-atan2(C,S))*invK;

    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}

__kernel void RSR (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax,
										__global double* ret)
{
  double C=cos((double)(*th0))-cos((double)(*th1));
  double S=2*(*_kmax)-sin((double)(*th0))+sin((double)(*th1));
  
  double temp1=2+4*pow2((*_kmax))-2*cos((double)(*th0)-(*th1))-4*(*_kmax)*(sin((double)(*th0))-sin((double)(*th1)));
  
  if (temp1<0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
    } else {
    double invK=1/(*_kmax);
    double sc_s1=mod2pi((*th0)-atan2(C,S))*invK;
    double sc_s2=invK*sqrt(temp1);
    double sc_s3=mod2pi(atan2(C,S)-(*th1))*invK;
    
    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}

__kernel void LSR (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax,
										__global double* ret)
{
  double C = cos((double)(*th0))+cos((double)(*th1));
  double S=2*(*_kmax)+sin((double)(*th0))+sin((double)(*th1));
  
  double temp1=-2+4*pow2((*_kmax))+2*cos((double)(*th0)-(*th1))+4*(*_kmax)*(sin((double)(*th0))+sin((double)(*th1)));
  if (temp1<0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
  } else {
    double invK=1/(*_kmax);
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1=mod2pi(atan2(-C,S)-atan2(-2, (*_kmax)*sc_s2)-(*th0))*invK;
    double sc_s3=mod2pi(atan2(-C,S)-atan2(-2, (*_kmax)*sc_s2)-(*th1))*invK;
    
    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}

__kernel void RSL (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax,
										__global double* ret)
{
  double C = cos((double)(*th0))+cos((double)(*th1));
  double S=2*(*_kmax)-sin((double)(*th0))-sin((double)(*th1));
  
  double temp1=-2+4*pow2((*_kmax))+2*cos((double)(*th0)-(*th1))-4*(*_kmax)*(sin((double)(*th0))+sin((double)(*th1)));
  if (temp1<0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
  } else {
    double invK=1/(*_kmax);
    
    double sc_s2=invK*sqrt(temp1);
    double sc_s1=mod2pi((*th0)-atan2(C,S)+atan2(2, (*_kmax)*sc_s2))*invK;
    double sc_s3=mod2pi((*th1)-atan2(C,S)+atan2(2, (*_kmax)*sc_s2))*invK;
    
    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}

__kernel void RLR (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax,
										__global double* ret)
{
  double C=cos((double)(*th0))-cos((double)(*th1));
  double S=2*(*_kmax)-sin((double)(*th0))+sin((double)(*th1));
  
  double temp1=0.125*(6-4*pow2((*_kmax))+2*cos((double)(*th0)-(*th1))+4*(*_kmax)*(sin((double)(*th0))-sin((double)(*th1))));
  
  if (fabs(temp1)-Epsi>1.0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
  } else {
    // if (equal(fabs(temp1), 1.0) ){
    //   temp1=round(temp1);
    // }
    
    double invK=1/(*_kmax);
    double sc_s2=mod2pi(2*M_PI-acos(temp1))*invK;
    double sc_s1=mod2pi((*th0)-atan2(C, S)+0.5*(*_kmax)*sc_s2)*invK;
    double sc_s3=mod2pi((*th0)-(*th1)+(*_kmax)*(sc_s2-sc_s1))*invK;
    
    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}

__kernel void LRL (	__global const double* th0, 
										__global const double* th1, 
										__global const double* _kmax,
										__global double* ret)
{
  double C=cos((double)(*th1))-cos((double)(*th0));
  double S=2*(*_kmax)+sin((double)(*th0))-sin((double)(*th1));
  
  double temp1=0.125*(6-4*pow2((*_kmax))+2*cos((double)(*th0)-(*th1))-4*(*_kmax)*(sin((double)(*th0))-sin((double)(*th1))));

  if (fabs(temp1)-Epsi>1.0){
    ret[0]=-1.0;
    ret[1]=-1.0;
    ret[2]=-1.0;
  } else {
    // if (equal(fabs(temp1), 1.0) ){
    //   temp1=round(temp1);
    // }

    double invK=1/(*_kmax);
    double sc_s2=mod2pi(2*M_PI-acos(temp1))*invK;
    double sc_s1=mod2pi(atan2(C, S)-(*th0)+0.5*(*_kmax)*sc_s2)*invK;
    double sc_s3=mod2pi((*th1)-(*th0)+(*_kmax)*(sc_s2-sc_s1))*invK;
    
    ret[0]=sc_s1;
    ret[1]=sc_s2; 
    ret[2]=sc_s3;
  }
}