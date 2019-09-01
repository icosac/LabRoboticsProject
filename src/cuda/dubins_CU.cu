#include <dubins_CU.hh>

#define pow2(x) x*x
#define CUDA_Epsi 1e-10
#define CUDA_DInf 0x7ff0000000000000 
#define CUDA_FInf 0x7f800000 

#define THREADS 256

// #define DUBINS_IN_KERNEL //Compute dubins using kernel. Doesn't need loop as the pair of points is computed through the id of the thread
#define DUBINS_IN_DEVICE //Compute dubins using devices. Needs to loop and call the function for each pair of points
// #define COMPUTE_ANGLES_BF_DUBINS //Compute angles before calling the deubins function
#define COMPUTE_ANGLES_IN_DUBINS //Compute angles inside the dubins function exploiting some maths. (reallyyyyyyyy slow)
#if defined DUBINS_IN_DEVICE && defined DUBINS_IN_KERNEL
#error DUBINS_IN_DEVICE and DUBINS_IN_KERNEL cannot be defined at the same time.
#endif
#if defined COMPUTE_ANGLES_IN_DUBINS && defined COMPUTE_ANGLES_BF_DUBINS
#error COMPUTE_ANGLES_BF_DUBINS and COMPUTE_ANGLES_IN_DUBINS cannot be defined at the same time.
#endif
// #if defined DUBINS_IN_DEVICE && defined COMPUTE_ANGLES_IN_DUBINS
// #error DUBINS_IN_DEVICE require COMPUTE_ANGLES_BF_DUBINS as the angles cannot be computed during the execution of the device function.
// #endif

#include<limits>
#include<iostream>
using namespace std;

#define DInf numeric_limits<double>::infinity()

__device__ bool CUDA_equal (double x, double y, double epsi=CUDA_Epsi){ return fabs(x-y)<epsi;}

__host__ double mod2pi (double angle){
	while(angle>2*M_PI || equal(angle, 2*M_PI)){
		angle-=(M_PI*2);
	}
	while(angle<0.0){
		angle+=(M_PI*2);
	}
	return angle;
}

__device__ double CUDA_mod2pi (double angle){
	while(angle>2*M_PI || CUDA_equal(angle, 2*M_PI)){
		angle-=(M_PI*2);
	}
	while(angle<0.0){
		angle+=(M_PI*2);
	}
	return angle;
}

__global__ void LSL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);
	double tan2=atan2(C, S);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s1=CUDA_mod2pi(tan2-th0)*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=CUDA_mod2pi(th1-tan2)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_LSL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__global__ void RSR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s1=CUDA_mod2pi(th0-atan2(C,S))*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=CUDA_mod2pi(atan2(C,S)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__global__ void LSR (double th0, double th1, double _kmax, double* ret, uint old=2)
{    
	double C = cos(th0)+cos(th1);
	double S=2*_kmax+sin(th0)+sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)+sin(th1));

	if (temp1<0){
	  ret[0]=-1;
		// printf("[%u] in_LSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f ret: %f\n", old, th0, th1, _kmax, C, S, temp1, ret[0]);
	  return;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= CUDA_mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th0)*invK;
	double sc_s3= CUDA_mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("[%u] in_LSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", old, th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__global__ void RSL (double th0, double th1, double _kmax, double* ret)
{
	double C = cos(th0)+cos(th1);
	double S=2*_kmax-sin(th0)-sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= CUDA_mod2pi(th0-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;
	double sc_s3= CUDA_mod2pi(th1-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RSL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__global__ void RLR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));

	if ((fabs(temp1)-CUDA_Epsi)>1.0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s2 = CUDA_mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = CUDA_mod2pi(th0-atan2(C, S)+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = CUDA_mod2pi(th0-th1+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RLR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__global__ void LRL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

	if ((fabs(temp1)-CUDA_Epsi)>1.0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s2 = CUDA_mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = CUDA_mod2pi(atan2(C, S)-th0+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = CUDA_mod2pi(th1-th0+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_LRL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

#ifdef DUBINS_IN_DEVICE
__device__ void __device__LSL (double th0, double th1, double _kmax, double* ret){
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);
	double tan2=atan2(C, S);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s1=CUDA_mod2pi(tan2-th0)*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=CUDA_mod2pi(th1-tan2)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_LSL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__device__ void __device__RSR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=2+4*pow2(_kmax)-2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1));

	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s1=CUDA_mod2pi(th0-atan2(C,S))*invK;
	double sc_s2=invK*sqrt(temp1);
	double sc_s3=CUDA_mod2pi(atan2(C,S)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__device__ void __device__LSR (double th0, double th1, double _kmax, double* ret, uint old=2)
{    
	double C = cos(th0)+cos(th1);
	double S=2*_kmax+sin(th0)+sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)+sin(th1));

	if (temp1<0){
	  ret[0]=-1;
		// printf("[%u] in_LSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f ret: %f\n", old, th0, th1, _kmax, C, S, temp1, ret[0]);
	  return;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= CUDA_mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th0)*invK;
	double sc_s3= CUDA_mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th1)*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("[%u] in_LSR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", old, th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__device__ void __device__RSL (double th0, double th1, double _kmax, double* ret)
{
	double C = cos(th0)+cos(th1);
	double S=2*_kmax-sin(th0)-sin(th1);

	double temp1=-2+4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)+sin(th1));
	if (temp1<0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= CUDA_mod2pi(th0-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;
	double sc_s3= CUDA_mod2pi(th1-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RSL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__device__ void __device__RLR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-CUDA_Epsi>1.0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s2 = CUDA_mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = CUDA_mod2pi(th0-atan2(C, S)+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = CUDA_mod2pi(th0-th1+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_RLR_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}

__device__ void __device__LRL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-CUDA_Epsi>1.0){
	  ret[0]=-1;
	  return;
	}

	double invK=1/_kmax;
	double sc_s2 = CUDA_mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = CUDA_mod2pi(atan2(C, S)-th0+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = CUDA_mod2pi(th1-th0+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	// printf("in_LRL_dev th0: %f, th1: %f kmax: %f C: %f S: %f temp1: %f invk: %f ret: %f %f %f\n", th0, th1, _kmax, C, S, temp1, invK, ret[0], ret[1], ret[2]);
}
#endif

static double sincCuda(double t) {
  if (std::abs(t)<0.002)
    return 1 - pow2(t)/6 * (1 - pow2(t)/20);
  else
    return sin(t)/t;
}

Configuration2<double> circlineCuda(double _L,
                                Configuration2<double> _P0,
                                double _K)
{
  double app=_K*_L/2.0;
  double sincc=_L*sincCuda(app);
  double phi=_P0.angle().toRad();
  
  double x=_P0.x() + sincc * cos(phi+app);
  double y=_P0.y() + sincc * sin(phi+app);
  Angle th=Angle(_K*_L+phi, Angle::RAD);	

  return Configuration2<double>(x, y, th);
}

class DubinsArcCuda {
public:
	double x0, y0, th0;
	double x1, y1, th1;
	double L, K;

	DubinsArcCuda(double _x0, double _y0, double _th0,
						double _x1, double _y1, double _th1, 
						double _L, double _K) : x0(_x0), y0(_y0), th0(_th0), 
						x1(_x1), y1(_y1), th1(_th1), L(_L), K(_K) {}

	DubinsArcCuda(Configuration2<double> start, double kmax, double _L) : L(_L), K(kmax) {
		x0=start.point().x();
		y0=start.point().y();
		th0=start.angle().toRad();
		Configuration2<double> end=circlineCuda(L, start, kmax);
		x1=end.point().x();
		y1=end.point().y();
		th1=end.angle().toRad();
	}

	void print() {
		printf("x0: %f, y0: %f, th0: %f, x1: %f, y1: %f, th1: %f, L: %f, K: %f\n", 
						x0, y0, th0, x1, y1, th1, L, K);
	}

	Configuration2<double> start(){
		return Configuration2<double> (x0, y0, Angle(th0, Angle::RAD));
	}

	Configuration2<double> end(){
		return Configuration2<double> (x1, y1, Angle(th1, Angle::RAD));
	}
};

void shortest_cuda(	double x0, double y0, double th0, 
										double x1, double y1, double th1, 
										double _kmax=1){
	double Length=DInf;
	int pidx=-1;
	double sc_s[3];
	
	//Scale to standard
	double phi=mod2pi(atan2((y1-y0), (x1-x0)));
	double lambda=sqrt(pow2((y1-y0))+pow2((x1-x0)))/2.0;
	double sc_th0=mod2pi(th0-phi);
	double sc_th1=mod2pi(th1-phi);
	double sc_Kmax=_kmax*lambda;

#ifdef STREAMS
	double** ret=(double**) malloc(sizeof(double*)*6);
	for(int i=0; i<6; i++){
		ret[i]=(double*) malloc(sizeof(double)*3);
	}
	double* dev_RSR; cudaMalloc((void**)&dev_RSR, 3*sizeof(double));
	double* dev_LSR; cudaMalloc((void**)&dev_LSR, 3*sizeof(double));
	double* dev_RSL; cudaMalloc((void**)&dev_RSL, 3*sizeof(double));
	double* dev_RLR; cudaMalloc((void**)&dev_RLR, 3*sizeof(double));
	double* dev_LRL; cudaMalloc((void**)&dev_LRL, 3*sizeof(double));
	double* dev_LSL; cudaMalloc((void**)&dev_LSL, 3*sizeof(double));

	cudaStream_t stream[6];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	cudaStreamCreate(&stream[3]);
	cudaStreamCreate(&stream[4]);
	cudaStreamCreate(&stream[5]);

	RSR<<<1, 1, 0, stream[0]>>>(sc_th0, sc_th1, sc_Kmax, dev_RSR);
	cudaMemcpyAsync(ret[0], dev_RSR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[0]);
	LSR<<<1, 1, 0, stream[1]>>>(sc_th0, sc_th1, sc_Kmax, dev_LSR);
	cudaMemcpyAsync(ret[1], dev_LSR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[1]);
	RSL<<<1, 1, 0, stream[2]>>>(sc_th0, sc_th1, sc_Kmax, dev_RSL);
	cudaMemcpyAsync(ret[2], dev_RSL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[2]);
	RLR<<<1, 1, 0, stream[3]>>>(sc_th0, sc_th1, sc_Kmax, dev_RLR);
	cudaMemcpyAsync(ret[3], dev_RLR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[3]);
	LRL<<<1, 1, 0, stream[4]>>>(sc_th0, sc_th1, sc_Kmax, dev_LRL);
	cudaMemcpyAsync(ret[4], dev_LRL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[4]);
	LSL<<<1, 1, 0, stream[5]>>>(sc_th0, sc_th1, sc_Kmax, dev_LSL);
	cudaMemcpyAsync(ret[5], dev_LSL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[5]);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);
	cudaStreamDestroy(stream[3]);
	cudaStreamDestroy(stream[4]);
	cudaStreamDestroy(stream[5]);

  cudaFree(dev_RSR);
	cudaFree(dev_LSR);
	cudaFree(dev_RSL);
	cudaFree(dev_RLR);
	cudaFree(dev_LRL);
	cudaFree(dev_LSL);

	for(int i=0; i<6; i++){
		double* value=ret[i];
		if (value[0]!=-1){
		  double appL=value[0]+value[1]+value[2];
		  if (appL<Length && !equal(appL, 0.0)){
		    Length = appL;
		    sc_s[0]=value[0];
		    sc_s[1]=value[1];
		    sc_s[2]=value[2];
		    pidx=i;
		  }
		}
  }

  for (int i=0; i<6; i++){
  	free(ret[i]);
  }
  free(ret);
#else
	double* ret=(double*) malloc(sizeof(double)*18);

	size_t pitch;
	double* dev_ret; cudaMallocPitch(&dev_ret, &pitch, 3*sizeof(double), 6);

	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret);
	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+1*pitch/sizeof(double));
	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+2*pitch/sizeof(double));
	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+3*pitch/sizeof(double));
	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+4*pitch/sizeof(double));
	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+5*pitch/sizeof(double));

	cudaMemcpy2D(ret, 3*sizeof(double), dev_ret, pitch, 3*sizeof(double), 6, cudaMemcpyDeviceToHost);

	cudaFree(dev_ret);

	// int a=0;
	// printf("RSR %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); a+=3;
	// printf("LSR %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); a+=3;
	// printf("RSL %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); a+=3;
	// printf("RLR %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); a+=3;
	// printf("LRL %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); a+=3;
	// printf("LSL %f %f %f\n", ret[a+0], ret[a+1], ret[a+2]); 

	for(int i=0; i<6; i++){
		double* value=ret+i*3;
		if (value[0]!=-1){
		  double appL=value[0]+value[1]+value[2];
		  if (appL<Length && !equal(appL, 0.0)){
		    Length = appL;
		    sc_s[0]=value[0];
		    sc_s[1]=value[1];
		    sc_s[2]=value[2];
		    pidx=i;
		  }
		}
  }
  if (pidx>=0){
  	//Scale back
  	double sc_std0=sc_s[0]*lambda;
  	double sc_std1=sc_s[1]*lambda;
  	double sc_std2=sc_s[2]*lambda;
  	double ksings[6][3]={
      { 1,  0,  1}, // LSL
      {-1,  0, -1}, // RSR
      { 1,  0, -1}, // LSR
      {-1,  0,  1}, // RSL
      {-1,  1, -1}, // RLR
      { 1, -1,  1}  // LRL
    };

    DubinsArcCuda A0 (Configuration2<double>(x0, y0, Angle(th0, Angle::RAD)), ksings[pidx][0]*_kmax, sc_std0);
    DubinsArcCuda A1 (A0.end(), ksings[pidx][1]*_kmax, sc_std1);
    DubinsArcCuda A2 (A1.end(), ksings[pidx][2]*_kmax, sc_std2);
    
    COUT(pidx)
    A0.print();
    A1.print();
    A2.print();
    cout << endl;
  }
  
  free(ret);

#endif
}


// __device__ __host__ 
// void toBase(double* v, const double* angles, const double* inc, 
// 						const int base, int value, size_t size, int startPos, int endPos){
// 	for (int i=0; i<size; i++){
// 		if (i<startPos || i>endPos){
// 			v[i]=angles[i];
// 		}
// 		else{
// 			v[i]=(value%base)*(*inc)+angles[i];
// 			value=(int)(value/base);
// 		}
// 	}
// }

__device__ __host__ 
void toBase(double* v, const double* angles, const double* inc, 
						const int base, int value, size_t size, int startPos, int endPos){
	for (int i=0; i<size; i++){
		if (i<startPos || i>endPos){
			v[i]=angles[i];
		}
		else{
			int rest=value%base;
			value=(int)(value/base);
			if (rest!=base-1){
				v[i]=(rest)*(*inc)+angles[i];			
			}
			else { //If special value then consider the two points on the same line.
				v[i]=angles[i-1];
			}
			#ifdef __CUDA_ARCH__
				v[i]=CUDA_mod2pi(v[i]);
			#else 
				v[i]=mod2pi(v[i]);
			#endif
		}
	}
}


#if __CUDA_ARCH__ < 600
__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}
#endif


#ifdef DUBINS_IN_KERNEL
#ifdef COMPUTE_ANGLES_IN_DUBINS
	__global__ void dubins (const double* x, const double* y, const double* th, double* length, const double* _kmax, 
													const size_t size, const double* inc, const size_t base, const uint old)
#else 
	__global__ void dubins (const double* x, const double* y, const double* th, double* length, const double* _kmax, const uint old)
#endif
#endif
#ifdef DUBINS_IN_DEVICE
#ifdef CUMPUTE_ANGLES_BF_DUBINS
	__device__ void dubins(	double* _x0, double* _y0, double _th0,
													double* _x1, double* _y1, double _th1, 
													double* _kmax, double* length, size_t old)
#else 
	__device__ void dubins(	double* _x0, double* _y0, double* _x1, double* _y1, 
													double* _th0, double* _th1, uint id, size_t base, double* inc, 
													size_t size, double* _kmax, double* length, size_t old)
#endif
#endif

{
	double kmax=*_kmax;

printf("[%u] [%u] %f %f\n", old, id, _th0[0], _th1[0]);
#ifdef DUBINS_IN_KERNEL
	uint tidx=threadIdx.x;
	
	double x0=x[tidx];
	double x1=x[tidx+1];
	double y0=y[tidx];
	double y1=y[tidx+1];
	double th0=th[tidx];
	double th1=th[tidx+1];
#endif
#ifdef DUBINS_IN_DEVICE
	double x0=_x0[0];
	double x1=_x1[0];
	double y0=_y0[0];
	double y1=_y1[0];
#ifdef COMPUTE_ANGLES_IN_DUBINS
	int tidx=id;
	double th0=_th0[0];
	double th1=_th1[0];
#else
	double th0=_th0;
	double th1=_th1;
#endif
#endif

printf("[%u] [%u], th0=%f, th1=%f, _th0=%f, _th1=%f\n", old, tidx, th0, th1, tidx, _th0[0], _th1[0]);

#ifdef COMPUTE_ANGLES_IN_DUBINS
	if (tidx!=(base-1)){
		int _pow=(int)(powf((double)base, (double)(tidx)));
		printf("[%u] [%u] %d\n", old, tidx, _pow);
		if (tidx==0){
			th1+=(*inc)*(((int)(old/_pow))%base); //_pow*base=_pow(base, tidx+1)
		}
		else if (tidx==(size-2)){
			th0+=(*inc)*(((int)((old*base)/_pow))%base); 
		}
		else if (old>=tidx){
			th0+=(*inc)*(((int)((old*base)/_pow))%base); 
			if (tidx<old){
				th1+=(*inc)*(((int)(old/_pow))%base); //_pow*base=_pow(base, tidx+1)
			}
		}
	}
	else { //Special case in which i check linearity
		th1=th0;
	}
	// printf("[%u] [%u], th0= %f, th1=%f, th[%u]=%f, th[%u]=%f\n", old, tidx, th0, th1, tidx, th[tidx], tidx+1, th[tidx+1]);
#endif

	//Scale to standard
	double sc_th0=CUDA_mod2pi(th0-atan2((y1-y0), (x1-x0)));
	double sc_th1=CUDA_mod2pi(th1-atan2((y1-y0), (x1-x0)));
	double sc_lambda=sqrt(pow2((y1-y0))+pow2((x1-x0)))/2.0;
	double sc_kmax=kmax*sc_lambda;

	double Length=CUDA_DInf;
	// printf("[%u] %f %f %f %f %f %f\ndy=%f dx=%f, atan2(dy, dx)=%f th0-atan2=%f mod2pi()=%f\nsc_th0: %f sc_th1: %f sc_lambda: %f sc_kmax: %f Length: %f\n", 
	// 			old, x0, y0, th0, x1, y1, th1, (y1-y0), (x1-x0), atan2((y1-y0),(x1-x0)), (th0-atan2((y1-y0), (x1-x0))), CUDA_mod2pi(th0-atan2((y1-y0), (x1-x0))), 
	// 		 	sc_th0, sc_th1, sc_lambda, sc_kmax, Length);

	double* ret=(double*) malloc(sizeof(double)*18);
	#ifdef DUBINS_IN_KERNEL
	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[0]));
	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[3]));
	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[6]), old);
	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[9]));
	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[12]));
	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, &(ret[15]));
	cudaError_t err=cudaDeviceSynchronize();
	if (err!=cudaSuccess){
		printf("Error: %d\n", err);
	}
	#elif defined DUBINS_IN_DEVICE
	__device__LSL(sc_th0, sc_th1, sc_kmax, &(ret[0]));
	__device__RSR(sc_th0, sc_th1, sc_kmax, &(ret[3]));
	__device__LSR(sc_th0, sc_th1, sc_kmax, &(ret[6]), old);
	__device__RSL(sc_th0, sc_th1, sc_kmax, &(ret[9]));
	__device__RLR(sc_th0, sc_th1, sc_kmax, &(ret[12]));
	__device__LRL(sc_th0, sc_th1, sc_kmax, &(ret[15]));
	#endif
	for (int i=0; i<6; i++){
		double* value=&ret[i*3];
		if (!CUDA_equal(value[0], -1.0)){
			double appL=(value[0]+value[1]+value[2])*sc_lambda;
			if (old==0){
				// printf("[%u] [%u] %d values %f %f %f\n", old, tidx, i, value[0], value[1], value[2]);
			}
			if (appL<Length){
				Length=appL;
			}
		}
		else {
			// printf("[%u] [%u] %d Nope %f %f %f\n", old, tidx, i, value[0], value[1], value[2]);
		}
	}
	if (Length<CUDA_DInf){
	  // printf("[%u] [%u] Length: %f {%f %f %f}\n", old, tidx, Length, ret[pidx*3], ret[pidx*3+1], ret[pidx*3+2]);
	  // printf("[%u] x0: %f y0: %f th0: %f x1: %f y1: %f th1: %f Length %f, length %f\n", old, x0, y0, th0, x1, y1, th1, Length, length[0]);
	  #if __CUDA_ARCH__ < 600 
	  	myAtomicAdd(length, Length);
  	#else 
		  atomicAdd(length, Length);
	  #endif
	  // printf("[%u] x0: %f y0: %f th0: %f x1: %f y1: %f th1: %f Length %f, length %f\n", old, x0, y0, th0, x1, y1, th1, Length, length[0]);
	}
	free(ret);
}

// __global__ void prova(double* x, double* y, double* angles, double* length, double* kmax){
// 	uint tidx=blockDim.x*blockIdx.x+threadIdx.x;
// 	angles[tidx]=100.0;
// 	angles[tidx+1]=100.0;
// 	x[tidx]=100.0;
// 	x[tidx+1]=100.0;
// 	y[tidx]=100.0;
// 	y[tidx+1]=100.0;
// 	MyatomicAdd(length, *kmax);
// }


__global__ void computeDubins (double* _angle, double* inc, double* x, double* y,
								double* lengths, ulong* dev_iter, size_t size, size_t base, double* _kmax, ulong* start){
	ulong tidx=blockDim.x*blockIdx.x+threadIdx.x;
	if (tidx>=(*dev_iter)){}
	else {
		ulong id=tidx;
		tidx+=(*start);
		#ifdef DUBINS_IN_KERNEL
		#ifdef COMPUTE_ANGLES_IN_DUBINS
			dubins<<<1, size-1>>>(x, y, _angle, lengths+tidx, _kmax, size, inc, base, tidx);
		#else //COMPUTE_ANGLES_BF_DUBINS
			double* angles=(double*) malloc(sizeof(double)*size);
			toBase(angles, _angle, inc, base, tidx, size, 1, size-2);
			// printf("[%d] inc: %f size %u\n", tidx, (*inc), size);
			// printf("[%d] angle+inc: %f, %f, %f, %f, %f\n", tidx, (angles[0]*(*inc)), (angles[1]*(*inc)), (angles[2]*(*inc)), (angles[3]*(*inc)), (angles[4]*(*inc)));
			// printf("[%d] angle: %f, %f, %f, %f, %f\n", tidx, (angles[0]), (angles[1]), (angles[2]), (angles[3]), (angles[4]));
			// printf("[%d] init: %f, %f, %f, %f, %f\n", tidx, _angle[0], _angle[1], _angle[2], _angle[3], _angle[4]);
			// printf("[%d] all: %f, %f, %f, %f, %f\n", tidx, (angles[0]*(*inc)+_angle[0]), (angles[1]*(*inc)+_angle[1]), (angles[2]*(*inc)+_angle[2]), (angles[3]*(*inc)+_angle[3]), (angles[4]*(*inc)+_angle[4]));
			// printf("[%d] _angle: %p inc: %p x: %p y: %p lengths: %p angles: %p \n", tidx, &(_angle[0]), inc, x, y, lengths, angles);
			
			// printf("[%u] angles[0]: %p, angles[1]: %p, angles[2]: %p, angles[3]: %p, angles[4]: %p\n", tidx, &angles[0], &angles[1], &angles[2], &angles[3], &angles[4]);
			// printf("[%u] angles[0]: %f, angles[1]: %f, angles[2]: %f, angles[3]: %f, angles[4]: %f\n", tidx, angles[0], angles[1], angles[2], angles[3], angles[4]);

			// prova<<<1, size-1>>> (x, y, angles, lengths+tidx, _skmax);
			dubins<<<1, size-1>>>(x, y, angles, lengths+tidx, _kmax, tidx);
			cudaDeviceSynchronize();
			
			// printf("[%u] angles[0]: %p, angles[1]: %p, angles[2]: %p, angles[3]: %p, angles[4]: %p\n", tidx, &angles[0], &angles[1], &angles[2], &angles[3], &angles[4]);
			// printf("[%u] angles[0]: %f, angles[1]: %f, angles[2]: %f, angles[3]: %f\n", tidx, angles[0], angles[1], angles[2], angles[3]);
			// printf("[%u] x[0]: %p, x[1]: %p, x[2]: %p, x[3]: %p, x[4]: %p\n", tidx, &x[0], &x[1], &x[2], &x[3], &x[4]);
			// printf("[%u] x[0]: %f, x[1]: %f, x[2]: %f, x[3]: %f, x[4]: %f\n", tidx, x[0], x[1], x[2], x[3], x[4]);
			// printf("[%u] y[0]: %p, y[1]: %p, y[2]: %p, y[3]: %p, y[4]: %p\n", tidx, &y[0], &y[1], &y[2], &y[3], &y[4]);
			// printf("[%u] y[0]: %f, y[1]: %f, y[2]: %f, y[3]: %f, y[4]: %f\n", tidx, y[0], y[1], y[2], y[3], y[4]);
			// printf("[%u] length: %f %p\n", tidx, lengths[tidx], &lengths[tidx]);
			free(angles);
		#endif
		#endif
		#ifdef DUBINS_IN_DEVICE			
		#ifdef COMPUTE_ANGLES_BF_DUBINS			
			double* angles=(double*) malloc(sizeof(double)*size);
			toBase(angles, _angle, inc, base, tidx, size, 1, size-2);
			// printf("[%u] angles[0]: %f, angles[1]: %f, angles[2]: %f, angles[3]: %f\n", tidx, angles[0], angles[1], angles[2], angles[3]);
			for (ulong i=0; i<size-1; i++){
				// printf("[%u] x: %f %f %f %f {%f %f}\n", tidx, x[0], x[1], x[2], x[3], (x+i)[0], (x+1+i)[0]);
				// printf("[%u] y: %f %f %f %f {%f %f}\n", tidx, y[0], y[1], y[2], y[3], (y+i)[0], (y+1+i)[0]);
				// dubins(x+i, y+i, angles[i], x+(i+1), y+(i+1), angles[i+1], _kmax, lengths+tidx, tidx); //When not splitting lenghts 
				dubins(x+i, y+i, angles[i], x+(i+1), y+(i+1), angles[i+1], _kmax, lengths+id, tidx);
				// printf("[%d] [%d] length: %f\n", tidx, i, lengths[tidx]);
			}
			// printf("[%u] %f\n", tidx, lengths[0]);
			free(angles);
		#else //COMPUTE_ANGLES_IN_DUBINS 
			printf("[%u] angles[0]: %f, angles[1]: %f, angles[2]: %f, angles[3]: %f\n", tidx, _angle[0], _angle[1], _angle[2], _angle[3]);
			for (uint i=0; i<size-1; i++){
				// printf("[%u] x: %f %f %f %f {%f %f}\n", tidx, x[0], x[1], x[2], x[3], (x+i)[0], (x+1+i)[0]);
				// printf("[%u] y: %f %f %f %f {%f %f}\n", tidx, y[0], y[1], y[2], y[3], (y+i)[0], (y+1+i)[0]);
				// dubins(x+i, y+i, angles[i], x+(i+1), y+(i+1), angles[i+1], _kmax, lengths+tidx, tidx); //When not splitting lenghts 
				printf("i: %u %f %f %f %f %f %f\n", i, *(x+i), *(y+i), *(x+(i+1)), *(y+(i+1)), *(_angle+i), *(_angle+(i+1)));
				dubins(x+i, y+i, x+(i+1), y+(i+1), _angle+i, _angle+(i+1), i, base, inc, size, _kmax, lengths+id, tidx);
				// printf("[%d] [%d] length: %f\n", tidx, i, lengths[tidx]);
			}
		#endif
		#endif
	}
}

#include<fstream>
#include <helper_cuda.h>

#define ITER_SIZE 1000000

double* dubinsSetBest(Configuration2<double> start,
										Configuration2<double> end,
										Tuple<Point2<double> > _points,
										int startPos,
										int endPos,
										uint parts, 
										double _kmax){
	size_t size=_points.size()+2;
	
	ofstream out_data; out_data.open("data/test/CUDA.test", fstream::app);

	unsigned long M=size-startPos;
  if (endPos>startPos){
    M-=(size-endPos-1);
  }
  parts++; //To consider points on the same line
  unsigned long iter_n=pow(parts, M);
  COUT(M)
  COUT(parts)
	COUT(size)
	cout << "Combinazioni: " << iter_n << endl;
	cout << "Iterazioni: " << iter_n << " " << ITER_SIZE << " " << (int)(iter_n/ITER_SIZE)+1 << endl;

	double* init_angle=(double*) malloc(sizeof(double)*size);
	double* x=(double*) malloc(size*sizeof(double));
	double* y=(double*) malloc(size*sizeof(double));

	init_angle[0]=start.angle().toRad();
	x[0]=start.point().x();
	y[0]=start.point().y();
	for (int i=1; i<size-2; i++){
		init_angle[i]=_points.get(i-1).th(_points.get(i)).toRad();
		x[i]=_points.get(i-1).x();
		y[i]=_points.get(i-1).y();
	}
	init_angle[size-2]=_points.get(_points.size()-1).th(end.point()).toRad();
	x[size-2]=_points.get(_points.size()-1).x();
	y[size-2]=_points.get(_points.size()-1).y();
	init_angle[size-1]=end.angle();
	x[size-1]=end.point().x();
	y[size-1]=end.point().y();
	
	for (int i=0; i<size; i++){
		cout << init_angle[i] << (i!=size-1 ? ", " : "\n");
	}

	double Length=DInf;
	double* lengths=(double*) malloc(sizeof(double)*(iter_n>ITER_SIZE ? ITER_SIZE : iter_n));
	double inc=2.0*M_PI/parts;

	auto start_t=Clock::now();

	double* dev_x; cudaMalloc((void**)&dev_x, sizeof(double)*size);
	double* dev_y; cudaMalloc((void**)&dev_y, sizeof(double)*size);
	double* dev_init_angle; cudaMalloc((void**)&dev_init_angle, sizeof(double)*size);

	double* dev_lengths; cudaMalloc((void**)&dev_lengths, sizeof(double)*(iter_n>ITER_SIZE ? ITER_SIZE : iter_n));// 
	// double* dev_lengths; cudaMalloc((void**)&dev_lengths, sizeof(double)*iter_n);
	// ulong* dev_iter; cudaMalloc((void**)&dev_iter, sizeof(ulong));
	double* dev_inc; cudaMalloc((void**)&dev_inc, sizeof(double)); 
	double* dev_kmax; cudaMalloc((void**)&dev_kmax, sizeof(double)); 
	auto stop=Clock::now();
	double elapsedMalloc=CHRONO::getElapsed(start_t, stop);
	// int* dev_pidxs; cudaMalloc((void**)&dev_pidxs, sizeof(int)*iter_n);
	start_t=Clock::now();
	cudaMemcpy(dev_x, x, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_init_angle, init_angle, sizeof(double)*size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_inc, &inc, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kmax, &_kmax, sizeof(double), cudaMemcpyHostToDevice);
	stop=Clock::now();
	double elapsedCopy=CHRONO::getElapsed(start_t, stop);

	COUT(elapsedMalloc)
	out_data << "elapsedMalloc: " << elapsedMalloc << endl;
	COUT(elapsedCopy)
	out_data << "elapsedCopy: " << elapsedCopy << endl;

	start_t=Clock::now();
	long long pidx=-1;
	int i, b;
	for (i=(int)(iter_n/ITER_SIZE); i>=0; i--) {
		printf("b: %u\n", b);
		ulong start=b*ITER_SIZE; b++;
		ulong n_iter = (i!=0) ? ITER_SIZE : (iter_n-start); 
		ulong* dev_iter; cudaMalloc(&dev_iter, sizeof(ulong));
		ulong* dev_start;cudaMalloc(&dev_start, sizeof(ulong));
		checkCudaErrors(cudaMemcpy(dev_iter, &n_iter, sizeof(ulong), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dev_start, &start, sizeof(ulong), cudaMemcpyHostToDevice));

		computeDubins<<<((int)(n_iter/THREADS)+1), (THREADS>n_iter ? n_iter : THREADS)>>> 
												(dev_init_angle, dev_inc, dev_x, dev_y, 
												dev_lengths, dev_iter, size, parts, dev_kmax, dev_start);
		
		checkCudaErrors(cudaMemcpy(lengths, dev_lengths, sizeof(double)*n_iter, cudaMemcpyDeviceToHost));

		for (int a=0; a<n_iter; a++){
			if (lengths[a]<Length && !equal(lengths[a], 0)){
				Length=lengths[a];
				pidx=a+start;
			}
		}
		checkCudaErrors(cudaFree(dev_lengths));
		checkCudaErrors(cudaMalloc(&dev_lengths, sizeof(double)*n_iter));

		checkCudaErrors(cudaFree(dev_iter));
		checkCudaErrors(cudaFree(dev_start));
	} 
	cout << "b: " << b << endl;

	int val=cudaDeviceSynchronize();
	stop=Clock::now();
	double elapsedCompute=CHRONO::getElapsed(start_t, stop);
	COUT(elapsedCompute)
	out_data << "elapsedCompute: " << elapsedCompute << endl;
	if (val!=cudaSuccess)
		printf("After dubins: %d\n", val);

	cudaMemcpy(lengths, dev_lengths, sizeof(double)*iter_n, cudaMemcpyDeviceToHost);

	// for (long long i=0; i<iter_n; i++){
	// 	// printf("lengths[%d] %f\n", i, lengths[i]); 
	// 	if (lengths[i]<Length && !equal(lengths[i], 0)){
	// 		Length=lengths[i];
	// 		pidx=i;
	// 	}
	// }

	COUT(Length)
	COUT(pidx)

	double* angls=(double*) malloc(sizeof(double)*size);
	toBase(angls, init_angle, &inc, parts, pidx, size, 1, size-2);
	for (int i=0; i<size; i++){
		cout << angls[i] << (i!=size-1 ? ", " : "\n");
	}

	for (int i=0; i<size-1; i++){
		shortest_cuda(x[i], y[i], angls[i], x[i+1], y[i+1], angls[i+1], _kmax);
	}

	free(lengths);

	cudaFree(dev_lengths);
	// cudaFree(dev_iter);
	cudaFree(dev_inc);
	return angls;
}
