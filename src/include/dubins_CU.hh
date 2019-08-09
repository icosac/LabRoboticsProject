#ifndef DUBINS_CU_HH
#define DUBINS_CU_HH
#ifdef CUDA

#include<maths.hh>

void shortest_cuda(	double sc_th0, double sc_th1, double sc_Kmax, 
					int& pidx, double* sc_s, double& Length);

void dubinsSetBest(	Configuration2<double> start,
					Configuration2<double> end,
					Tuple<Point2<double> > _points,
					int startPos,
					int endPos,
					uint parts, 
					double _kmax=1.0);

#endif
#endif