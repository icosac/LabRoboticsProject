#ifdef CUDA
#ifndef DUBINS_CU_HH
#define DUBINS_CU_HH

#include<utils.hh>

void shortest_cuda(	double sc_th0, double sc_th1, double sc_Kmax, 
					int& pidx, double* sc_s, double& Length);

#endif
#endif