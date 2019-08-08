#include <dubins_CU.hh>

#define pow2(x) x*x
#define CUDA_Epsi 1e-10
#define CUDA_DInf 0x7ff0000000000000 
#define CUDA_FInf 0x7f800000 

#include<limits>
#include<iostream>
using namespace std;

#define DInf numeric_limits<double>::infinity()

__device__ __host__ bool equal (double x, double y, double epsi=CUDA_Epsi){ return fabs(x-y)<epsi;}

__device__ double mod2pi (double angle){
	printf("mod2pi\n");
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


#define GRID 1
#define THREADS 256
//TODO test implementation where x_i=y%base^i
__device__ void toBase(double* v, const int base, int value){
	int i=0;
	while(value>0){
		v[i]=value%base;
		value=(int)(value/base);
		i++;
	}
}

// typedef struct {
// 	double x0, y0, th0;
// 	double x1, y1, th1;
// 	double L, K;

// 	__host__ __device__ dubinsArc(double _x0, double _y0, double _th0,
// 						double _x1, double _y1, double _th1, 
// 						double _L, double _K) : x0(_x0), y0(_y0), th0(_th0), 
// 						x1(_x1), y1(_y1), th1(_th1), L(_L), K(_K) {}

// 	print() {
// 		printf("x0: %f, y0: %f, th0: %f, x1: %f, y1: %f, th1: %f, L: %f, K: %f\n", 
// 						x0, y0, th0, x1, y1, th1, L, K);
// 	}

// } dubinsArc;


// __device__ void dubins(	double* x0, double* y0, double th0,
// 													double* x1, double* y1, double th1, 
// 													double _kmax, double* L, int* id){
// 	int pidx=-1;
// 	//Scale to standard
// 	printf("x0: %f\n", *x0);
// 	printf("atan2: %f\n", atan2(((*y1)-(*y0)), ((*x1)-(*x0))));
// 	double sc_th0=mod2pi((th0)-atan2(((*y1)-(*y0)), ((*x1)-(*x0))));
// 	printf("ciao\n");
// 	double sc_th1=mod2pi((th1)-atan2(((*y1)-(*y0)), ((*x1)-(*x0))));
// 	double sc_lambda=sqrt(pow2((*y1)-(*y0))+pow2((*x1)-(*x0)))/2;
// 	double sc_kmax=_kmax*sc_lambda;

// 	double Length=CUDA_DInf;
// 	// double sc_s1=0.0;
// 	// double sc_s2=0.0;
// 	// double sc_s3=0.0;

// 	double* ret=new double [18];

// 	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret);
// 	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+3*sizeof(double));
// 	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+6*sizeof(double));
// 	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+9*sizeof(double));
// 	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+12*sizeof(double));
// 	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+15*sizeof(double));
// 	printf("%f\n", ret[0]);
// 	printf("%f\n", ret[1]);
// 	printf("%f\n", ret[2]);

// 	__syncthreads();

// 	for (int i=0; i<6; i++){
// 		double* value=ret+i*3*sizeof(double);
// 		if (value[0]!=-1){
// 			double appL=value[0]+value[1]+value[2];
// 			if (appL<Length){
// 				Length=appL;
// 				// sc_s1=value[0];
// 				// sc_s1=value[1];
// 				// sc_s1=value[2];
// 				pidx=i;
// 			}
// 		}
// 	}
// 	printf("ciao:\n");
// 	printf("%f\n", Length);
// 	*L=Length;
// 	*id=pidx;

	// if (pidx>=0){
	// 	//Scale from standard
	// 	sc_s1*=sc_lambda;
	// 	sc_s2*=sc_lambda;
	// 	sc_s3*=sc_lambda;

	// 	int ksings[6][3]={
	//      { 1,  0,  1}, // LSL
	//      {-1,  0, -1}, // RSR
	//      { 1,  0, -1}, // LSR
	//      {-1,  0,  1}, // RSL
	//      {-1,  1, -1}, // RLR
	//      { 1, -1,  1}  // LRL
	//    };

	//    #define L sc_s1
	//    double K=_kmax*ksings[pidx][0];
	//    double app=K*L/2.0;
	//    double sincc=L*sinc(app);
	//    dubinsArc A1(	x0, y0, th0, L, 
	//    						(x0+sincc*cos(app)), (y0+sincc+sin(app)),
	//    						mod2pi(K*L+th0), K);
	  
	  
	//    #define L sc_s2
	//    K=_kmax*ksings[pidx][1];
	//    app=K*L/2.0;
	//    sincc=L*sinc(app);
	//    dubinsArc A1(	A0.x1, A0.y1, A0.th1, 
	//    							(A0.x1+sincc*cos(app)), (A0.y1+sincc+sin(app)),
	//    							mod2pi(_kmax*L+A0.th1), 
	//    							L, ksings[pidx][0]*_kmax);

	//   	#define L sc_s3
	//    K=_kmax*ksings[pidx][3];
	//    app=K*L/2.0;
	//    sincc=L*sinc(app);
	//    dubinsArc A2(	A1.x1, A1.y1, A1.th1, 
	//    							(A1.x1+sincc*cos(app)), (A1.y1+sincc+sin(app)),
	//    							mod2pi(_kmax*L+A1.th1), 
	//    							L, ksings[pidx][0]*_kmax); 
	// }
// }

#if __CUDA_ARCH__ < 600
__device__ double MyatomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void dubins (const double* x, const double* y, double* th, double* length, double _kmax){
	uint tidx=threadIdx.x;
	printf("ciao %u\n", tidx);
	double x0=x[tidx];
	double x1=x[tidx+1];
	double y0=y[tidx];
	double y1=y[tidx+1];
	double th0=th[tidx];
	double th1=th[tidx+1];

	printf("ciao:\n");
	int pidx=-1;
	//Scale to standard
	printf("x0: %f\n", x0);
	printf("atan2: %f\n", atan2((y1-y0), (x1-x0)));
	double sc_th0=mod2pi(th0-atan2((y1-y0), (x1-x0)));
	printf("ciao\n");
	double sc_th1=mod2pi(th1-atan2((y1-y0), (x1-x0)));
	double sc_lambda=sqrt(pow2(y1-y0)+pow2(x1-x0))/2;
	double sc_kmax=_kmax*sc_lambda;

	double Length=CUDA_DInf;
	// double sc_s1=0.0;
	// double sc_s2=0.0;
	// double sc_s3=0.0;

	double* ret=new double [18];

	LSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret);
	RSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+3*sizeof(double));
	LSR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+6*sizeof(double));
	RSL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+9*sizeof(double));
	RLR<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+12*sizeof(double));
	LRL<<<1, 1>>>(sc_th0, sc_th1, sc_kmax, ret+15*sizeof(double));
	printf("%f\n", ret[0]);
	printf("%f\n", ret[1]);
	printf("%f\n", ret[2]);

	__syncthreads();

	for (int i=0; i<6; i++){
		double* value=ret+i*3*sizeof(double);
		if (value[0]!=-1){
			double appL=value[0]+value[1]+value[2];
			if (appL<Length){
				Length=appL;
				// sc_s1=value[0];
				// sc_s1=value[1];
				// sc_s1=value[2];
				pidx=i;
			}
		}
	}
	printf("ciao:\n");
	printf("%f\n", Length);
	MyatomicAdd(length, Length);

}

__global__ void computeDubins (const double* _angle, const double* inc, const double* x, const double* y,
															double* lengths, int* pidxs, uint dev_iter, size_t size, size_t base, double _kmax){
	uint tidx=blockDim.x*blockIdx.x+threadIdx.x;
	if (tidx>dev_iter){}
	else {
		double* angles=(double*) malloc(sizeof(double)*size);
		toBase(angles, base, tidx);
		// dubins<<<GRID, THREADS>>>(x, y, angle, &(lengths[tidx]), _kmax);
		printf("[%d] inc: %f\n", tidx, (*inc));
		printf("[%d] angle+inc: %f, %f, %f, %f, %f\n", tidx, (angles[0]*(*inc)), (angles[1]*(*inc)), (angles[2]*(*inc)), (angles[3]*(*inc)), (angles[4]*(*inc)));
		printf("[%d] init: %f, %f, %f, %f, %f\n", tidx, _angle[0], _angle[1], _angle[2], _angle[3], _angle[4]);
		printf("[%d] all: %f, %f, %f, %f, %f\n", tidx, (angles[0]*(*inc)+_angle[0]), (angles[1]*(*inc)+_angle[1]), (angles[2]*(*inc)+_angle[2]), (angles[3]*(*inc)+_angle[3]), (angles[4]*(*inc)+_angle[4]));
		dubins<<<1, size-1>>>(x, y, angles, &lengths[tidx], _kmax);
		// for (int i=0; i<size-1; i++){
		// 	dubins(x+i*sizeof(double), y+i*sizeof(double), angles[i]*(*inc)+_angle[i], 
		// 				x+(i+1)*sizeof(double), y+(i+1)*sizeof(double), angles[i+1]*(*inc)+_angle[i+1], 
		// 				_kmax, &lengths[tidx], &pidxs[tidx]);
		// 	printf("[%d] [%d] length: %f\n", tidx, i, lengths[tidx]);
		// }
		free(angles);
	}
}

void dubinsSetBest(Configuration2<double> start,
										Configuration2<double> end,
										Tuple<Point2<double> > _points,
										uint parts, 
										double _kmax){
	size_t size=_points.size()+2;
	COUT(parts)
	COUT(size)
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

	double Length=0.0;
	double* lengths=(double*) malloc(sizeof(double)*iter_n);
	int* pidxs=(int*) malloc(sizeof(int)*iter_n);
	double inc=2.0*M_PI/parts;

	double* dev_init_angle; cudaMalloc((void**)&dev_init_angle, sizeof(double)*size);
	double* dev_lengths; cudaMalloc((void**)&dev_lengths, sizeof(double)*iter_n); 
	uint* dev_iter; cudaMalloc((void**)&dev_iter, sizeof(uint));
	double* dev_inc; cudaMalloc((void**)&dev_inc, sizeof(double)); 
	int* dev_pidxs; cudaMalloc((void**)&dev_pidxs, sizeof(int)*iter_n);

	cudaMemcpy(dev_init_angle, init_angle, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_iter, &iter_n, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inc, &inc, sizeof(double), cudaMemcpyHostToDevice);

	// computeDubins<<<((int)(iter_n/THREADS)+1), (THREADS>iter_n ? iter_n : THREADS)>>> 
	computeDubins<<<1, 10>>> 
																		(dev_init_angle, dev_inc, x, y, 
																		dev_lengths, dev_pidxs, iter_n, 
																		size, parts, _kmax);
	cudaDeviceSynchronize();
	cudaMemcpy(lengths, dev_lengths, sizeof(double)*iter_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(pidxs, dev_pidxs, sizeof(double)*iter_n, cudaMemcpyDeviceToHost);

	int pidx;

	for (int i=0; i<iter_n; i++){
		// COUT(lengths[i])
		// COUT(pidxs[i])
		if (lengths[i]<Length){
			Length=lengths[i];
			pidx=pidxs[i];
		}
	}

	COUT(Length)
	COUT(pidx)

	free(pidxs);
	free(lengths);

	cudaFree(dev_lengths);
	cudaFree(dev_pidxs);
	cudaFree(dev_iter);
	cudaFree(dev_inc);
}