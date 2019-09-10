#ifndef PLANNING_HH
#define PLANNING_HH

#include <iostream>
#include <vector>
#include <utility>

#include <map.hh>
#include <maths.hh>
#include <settings.hh>
#include <objects.hh>
#include <detection.hh>
#include <dubins.hh>
#include "path.h"

using namespace std;
using namespace cv;

namespace Planning {
	extern Mapp* map;

	vector<Point2<int> > planning(const Mat & img);
	void createMapp();

	void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn);
	void loadVP(vector<Point2<int> > & vp, FileNode fn);

	void fromVpToPath(vector<Point2<int> > & vp, Path & path);

	// template<class T>
	void plan_best(const Configuration2<double>& _start, vector<vector<Point2<int> > >& vPoints);
}



#endif