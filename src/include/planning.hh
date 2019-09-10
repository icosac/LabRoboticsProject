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
	extern Configuration2<double> conf;

	constexpr static double baseDistance = -1.0;  //It is the reference base distance for the matrix of distances
    const int range = 3;            // It is the foundamental parameter of the function minPath (the right compromise its 3)
    const int foundLimit = 20;       // Empiric limit of found, it represent how many times the destination will be visited before the end of the BFS. 0 is the base case (first visit=stop) ~150, or better, none is the opposite limit.
    static const int nPoints = 50;  // It is the number of points that the function sampleNPoints will sample from the computed vector of vector retrieved from the minPath.    

	vector<Point2<int> > planning(const Mat & img);
	void createMapp();

	void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn);
	void loadVP(vector<Point2<int> > & vp, FileNode fn);

    vector<vector<Point2<int> > > minPathNPointsWithChoice(const vector<Point2<int> > & vp, const double bonus);
    vector<vector<Point2<int> > > minPathNPoints(const vector<Point2<int> > & vp);
    vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1);
	    vector<Point2<int> > minPathTwoPointsInternal(
	                            const Point2<int> & startP, const Point2<int> & endP, 
	                            double ** distances, Point2<int> ** parents,
	                            const bool firstSegment=false);

    	void intToVect(int c, vector<int> & v);
	    void resetDistanceMap(double ** distances, const double value = baseDistance);
    
    vector<Point2<int> > sampleNPoints(const vector<vector<Point2<int> > > & vvp, const int n=nPoints);
        vector<Point2<int> > sampleNPoints(const vector<Point2<int> > & points, const int n);
        vector<Point2<int> > samplePointsEachNCells(const vector<Point2<int> > & points, const int step);

	void fromVpToPath(vector<Point2<int> > & vp, Path & path);
    int getNPoints();

	// template<class T>
	void plan_best(const Configuration2<double>& _start, vector<vector<Point2<int> > >& vPoints);
}



#endif