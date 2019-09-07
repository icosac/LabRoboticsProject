#include "dubins.hh"

/* Nothing to see here */

//TODO should I keep this or not?
// #define DELTA M_PI/180.0 //1 degree

// template <class T>
// vector<Point2<T> > reduce_points(Tuple<Point2<T> > init_points){
//   double delta=DELTA;
//   vector<Point2<T> > ret={};
//   for (int i=0; i<init_points.size(); i++){
//     Point2<T> app=init_points.get(i);
//     if (i==0 || i==init_points.size()-1){
//       ret.push_back(app);
//     }
//     else {
//       if (ret.back().th(app).toRad()<delta){
//         delta+=DELTA;
//       }
//       else {
//         ret.push_back(app);
//         delta=DELTA;
//       }
//     }
//   }
//   return ret;
// }

// template<class T>
// Dubins<T> start_pos (const Configuration2<T> _start, 
// 												const vector<Point2<T> >& vPoints){
// 	Dubins<T> start_dub;
// 	int i=0;
// 	while (start_dub.pidx()<0 && i<vPoints.size()-1){
// 		start_dub=Dubins(_start, Configuration2<T>(vPoints[i], vPoints[i].th(vPoints[i+1])) );
// 	}
// 	#ifdef DEBUG
// 	if (start_dub.pidx()>=0){
// 		COUT(start_dub)
// 	}
// 	#endif
// 	//TODO Add check for obstacles?
// }

// template<class T>
// vector<Point2<T> > plan_best(const Configuration2<T> _start,
// 																vector<Point2<T> > vPoints){
// 	start_pos(_start, vPoints);
// }