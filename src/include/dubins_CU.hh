#ifndef DUBINS_CU_HH
#define DUBINS_CU_HH
#ifdef CUDA

#include<maths.hh>

namespace My_CUDA{
void shortest_cuda(	double sc_th0, double sc_th1, double sc_Kmax, 
					int& pidx, double* sc_s, double& Length);

double* dubinsSetBest(Configuration2<double> start,
											Configuration2<double> end,
											Tuple<Point2<double> > _points,
											int startPos,
											int endPos,
											ulong iter_n,
											double inc,
											size_t parts, 
											double _kmax);
}
// double* dubinsSetCuda(Configuration2<double> start,
// 											Configuration2<double> end,
// 											Tuple<Point2<double> > _points,
// 											double _kmax=1,
// 											int startPos=0,
// 											int endPos=-1,
// 											uint parts=2 );

#endif
#endif