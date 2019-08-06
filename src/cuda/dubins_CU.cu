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
}

// void shortest_cuda(	double x0, double y0, double th0, 
// 										double x1, double y1, double th1, 
// 										double sc_Kmax, int& pidx, double* sc_s, double& Length){
// 	Length=DInf;
// 	double sc_s1=0.0;
// 	double sc_s2=0.0;
// 	double sc_s3=0.0;
	
// #ifdef STREAMS
// 	double** ret=(double**) malloc(sizeof(double*)*6);
// 	for(int i=0; i<6; i++){
// 		ret[i]=(double*) malloc(sizeof(double)*3);
// 	}
// 	double* dev_RSR; cudaMalloc((void**)&dev_RSR, 3*sizeof(double));
// 	double* dev_LSR; cudaMalloc((void**)&dev_LSR, 3*sizeof(double));
// 	double* dev_RSL; cudaMalloc((void**)&dev_RSL, 3*sizeof(double));
// 	double* dev_RLR; cudaMalloc((void**)&dev_RLR, 3*sizeof(double));
// 	double* dev_LRL; cudaMalloc((void**)&dev_LRL, 3*sizeof(double));
// 	double* dev_LSL; cudaMalloc((void**)&dev_LSL, 3*sizeof(double));

// 	cudaStream_t stream[6];
// 	cudaStreamCreate(&stream[0]);
// 	cudaStreamCreate(&stream[1]);
// 	cudaStreamCreate(&stream[2]);
// 	cudaStreamCreate(&stream[3]);
// 	cudaStreamCreate(&stream[4]);
// 	cudaStreamCreate(&stream[5]);

// 	RSR<<<1, 1, 0, stream[0]>>>(sc_th0, sc_th1, sc_Kmax, dev_RSR);
// 	cudaMemcpyAsync(ret[0], dev_RSR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[0]);
// 	LSR<<<1, 1, 0, stream[1]>>>(sc_th0, sc_th1, sc_Kmax, dev_LSR);
// 	cudaMemcpyAsync(ret[1], dev_LSR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[1]);
// 	RSL<<<1, 1, 0, stream[2]>>>(sc_th0, sc_th1, sc_Kmax, dev_RSL);
// 	cudaMemcpyAsync(ret[2], dev_RSL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[2]);
// 	RLR<<<1, 1, 0, stream[3]>>>(sc_th0, sc_th1, sc_Kmax, dev_RLR);
// 	cudaMemcpyAsync(ret[3], dev_RLR, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[3]);
// 	LRL<<<1, 1, 0, stream[4]>>>(sc_th0, sc_th1, sc_Kmax, dev_LRL);
// 	cudaMemcpyAsync(ret[4], dev_LRL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[4]);
// 	LSL<<<1, 1, 0, stream[5]>>>(sc_th0, sc_th1, sc_Kmax, dev_LSL);
// 	cudaMemcpyAsync(ret[5], dev_LSL, sizeof(double)*3, cudaMemcpyDeviceToHost, stream[5]);

// 	cudaStreamDestroy(stream[0]);
// 	cudaStreamDestroy(stream[1]);
// 	cudaStreamDestroy(stream[2]);
// 	cudaStreamDestroy(stream[3]);
// 	cudaStreamDestroy(stream[4]);
// 	cudaStreamDestroy(stream[5]);

//   cudaFree(dev_RSR);
// 	cudaFree(dev_LSR);
// 	cudaFree(dev_RSL);
// 	cudaFree(dev_RLR);
// 	cudaFree(dev_LRL);
// 	cudaFree(dev_LSL);

// 	for(int i=0; i<6; i++){
// 		double* value=ret[i];
// 		if (value[0]!=-1){
// 		  double appL=value[0]+value[1]+value[2];
// 		  if (appL<Length && !equal(appL, 0.0)){
// 		    Length = appL;
// 		    sc_s[0]=value[0];
// 		    sc_s[1]=value[1];
// 		    sc_s[2]=value[2];
// 		    pidx=i;
// 		  }
// 		}
//   }

//   for (int i=0; i<6; i++){
//   	free(ret[i]);
//   }
//   free(ret);

// #else

// 	double* ret=(double*) malloc(sizeof(double)*18);

// 	size_t pitch;
// 	double* dev_ret; cudaMallocPitch(&dev_ret, &pitch, 3*sizeof(double), 6);

// 	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret);
// 	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+1*pitch/sizeof(double));
// 	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+2*pitch/sizeof(double));
// 	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+3*pitch/sizeof(double));
// 	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+4*pitch/sizeof(double));
// 	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_Kmax, dev_ret+5*pitch/sizeof(double));

// 	cudaMemcpy2D(ret, 3*sizeof(double), dev_ret, pitch, 3*sizeof(double), 6, cudaMemcpyDeviceToHost);

// 	cudaFree(dev_ret);

// 	for(int i=0; i<6; i++){
// 		double* value=ret+i*3;
// 		if (value[0]!=-1){
// 		  double appL=value[0]+value[1]+value[2];
// 		  if (appL<Length && !equal(appL, 0.0)){
// 		    Length = appL;
// 		    sc_s[0]=value[0];
// 		    sc_s[1]=value[1];
// 		    sc_s[2]=value[2];
// 		    pidx=i;
// 		  }
// 		}
//   }
  
//   free(ret);

// #endif
// }

//TODO test implementation where x_i=y%base^i
__device__ void toBase(double* v, const int base, int value){
	int i=0;
	while(value>0){
		v[i]=value%base;
		value=(int)(val/BASE);
		i++;
	}
}

__global__ void dubins()

__global__ void computeDubins (const double* _angle, const double* inc; const double* x, const double* y,
															double* lengths, uint dev_iter, size_t size, double _kmax){
	uint pidx=blockDim.x*blockIdx.x+threadIdx.x;
	double* angles=(double*) malloc(sizeof(double)*size);
	if (pidx>dev_iter){}
	else {
		toBase(angle, base, pidx);
		double angle=0;
		dubins<<<GRID, THREADS>>>(x, y, angle, &(lenghts[pidx]), _kmax);

		for (int i=0; i<size-1; i++){
			angles[i]=angles[i]*(*inc)+_angle[i];
			angles[i+1]=angles[i+1]*(*inc)+_angle[i+1];
			dubins(x[i], y[i], angle[i], x[i+1], y[i+1], angle[i+1], _kmax);
			angle=angle[i];
		}
	}

}

void dubinsSetBest(Configuration2<double> start,
										Configuration2<double> end,
										Tuple<Point2<double> > _points
										uint parts){
	int size=_points.size()+2;
	unsigned long iter_n=pow(parts, size);
	COUT(iter_n)

	double* init_angle=(double*) malloc(sizeof(double)*size);
	double* x=(double*) malloc(size*sizeof(double));
	double* y=(double*) malloc(size*sizeof(double));

	init_angle[0]=start.angle().toRad();
	x[0]=start.point().x();
	y[0]=start.point().y();
	for (int i=1; i<size-2; i++){
		init_angle[i]=_points.get(i-1).th(_points.get(i)).toRad();
		x[0]=_points.get(i-1).x();
		y[0]=_points.get(i-1).y();
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

	double Lenght=0.0;
	double* lenghts=(double*) malloc(sizeof(double)iter_n);
	double* dev_lengths; cudaMalloc((void**)&dev_lengths, sizeof(double)*iter_n); 

	uint* dev_iter; cudaMalloc((void**)&dev_ret, sizeof(uint));
	cudaMemcpy(dev_iter, &iter_n, sizeof(uint), cudaMemcpyHostToDevice);

	computeDubins<<<GRID, THREADS>>> (init_angle, x, y, dev_lengths, dev_iter);

	cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)

}