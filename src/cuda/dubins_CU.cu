#include <dubins_CU.hh>

#define pow2(x) x*x
#define CUDA_Epsi 1e-10

#include<limits>
#include<iostream>
using namespace std;

#define DInf numeric_limits<double>::infinity()

__device__ __host__ bool equal (double x, double y, double epsi=CUDA_Epsi){ return fabs(x-y)<epsi;}

__device__ double mod2pi (double angle){
	while(angle>=2*M_PI){
		angle-=(M_PI*2);
	}
	while(angle<0){
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
	  ret[0]=-1;
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
	  ret[0]=-1;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th0)*invK;
	double sc_s3= mod2pi(atan2(-C,S)-atan2(-2.0, _kmax*sc_s2)-th1)*invK;

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
	  ret[0]=-1;
	}

	double invK=1/_kmax;

	double sc_s2=invK*sqrt(temp1);
	double sc_s1= mod2pi(th0-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;
	double sc_s3= mod2pi(th1-atan2(C,S)+atan2(2.0, _kmax*sc_s2))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
}

__global__ void RLR (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th0)-cos(th1);
	double S=2*_kmax-sin(th0)+sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)+4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-CUDA_Epsi>1.0){
	  ret[0]=-1;
	}

	double invK=1/_kmax;
	double sc_s2 = mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = mod2pi(th0-atan2(C, S)+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = mod2pi(th0-th1+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;
	if (th0==1.2 && th1==1.2){
		printf("RLR 0: %f\n", ret[0]);
		printf("RLR 1: %f\n", ret[1]);
		printf("RLR 2: %f\n", ret[2]);
	}
}

__global__ void LRL (double th0, double th1, double _kmax, double* ret)
{
	double C=cos(th1)-cos(th0);
	double S=2*_kmax+sin(th0)-sin(th1);

	double temp1=0.125*(6-4*pow2(_kmax)+2*cos(th0-th1)-4*_kmax*(sin(th0)-sin(th1)));

	if (fabs(temp1)-CUDA_Epsi>1.0){
	  ret[0]=-1;
	}

	double invK=1/_kmax;
	double sc_s2 = mod2pi(2*M_PI-acos(temp1))*invK;
	double sc_s1 = mod2pi(atan2(C, S)-th0+0.5*_kmax*sc_s2)*invK;
	double sc_s3 = mod2pi(th1-th0+_kmax*(sc_s2-sc_s1))*invK;

	ret[0]=sc_s1;
	ret[1]=sc_s2;
	ret[2]=sc_s3;

	if (th0==1.2 && th1==1.2){
		printf("LRL 0: %f\n", ret[0]);
		printf("LRL 1: %f\n", ret[1]);
		printf("LRL 2: %f\n", ret[2]);
	}
}

void shortest_cuda(	double sc_th0, double sc_th1, double sc_Kmax, 
										int& pidx, double* sc_s, double& Length){
	Length=DInf;
	double sc_s1=0.0;
	double sc_s2=0.0;
	double sc_s3=0.0;
	
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

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_RSR);
	cudaMemcpyAsync(ret[0], dev_RSR, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);
	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_LSR);
	cudaMemcpyAsync(ret[1], dev_LSR, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);
	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_RSL);
	cudaMemcpyAsync(ret[2], dev_RSL, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);
	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_RLR);
	cudaMemcpyAsync(ret[3], dev_RLR, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);
	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_LRL);
	cudaMemcpyAsync(ret[4], dev_LRL, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);
	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_LSL);
	cudaMemcpyAsync(ret[5], dev_LSL, sizeof(double)*3, cudaMemcpyDeviceToHost, 0);

	cudaStreamDestroy(stream);

	// for(size_t i=0; i<6; ++i){
	// 	cudaMemcpy(&ret[i*3], &dev_ret[pitch*i], 6*sizeof(double), cudaMemcpyDeviceToHost);
	// }

	// cudaMemcpy2D(ret, pitch_h, dev_ret, pitch, 3*sizeof(double), 6, cudaMemcpyDeviceToHost);

	for(int i=0; i<6; i++){
		// double* value=ret+i*3;
		double* value=ret[i];
		if (ret[i][0]!=-1){
		  double appL=value[0]+value[1]+value[2];
		  // double appL=ret[i][0]+ret[i][1]+ret[i][2];
		  if (appL<Length && !equal(appL, 0.0)){
		    Length = appL;
		    sc_s[0]=value[0];
		    sc_s[1]=value[1];
		    sc_s[2]=value[2];
		    pidx=i;
		  }
		}
  }

  cudaFree(dev_RSR);
	cudaFree(dev_LSR);
	cudaFree(dev_RSL);
	cudaFree(dev_RLR);
	cudaFree(dev_LRL);
	cudaFree(dev_LSL);

  for (int i=0; i<6; i++){
  	free(ret[i]);
  }
  free(ret);
}

// int main (){
// 	int pidx=-1;
// 	shortest_cuda(0.0, 0.0, 0.0, pidx);
// 	cout << pidx << endl;
// 	return 0;
// }