#ifndef PLANNING_HH
#define PLANNING_HH

#include <iostream>
#include <vector>
#include <utility>
#include <iomanip>

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
	constexpr static double baseDir = -1.0;  //It is the reference base direction for the angle sampling in the minPath
    extern const int nAngles;
    extern const double angleRange; // It is a range that delimit the possibilities of the min path of adding new elements. Practically is used to create a path, more or less, straight that has at most an angle between two consecutive segments of angleRange.

    extern const int range;            // It is the foundamental parameter of the function minPath (the right compromise its 3)
    const int foundLimit = 20;       // Empiric limit of found, it represent how many times the destination will be visited before the end of the BFS. 0 is the base case (first visit=stop) ~150, or better, none is the opposite limit.
    const int foundLimitAngles = 40;       // Empiric limit of found, it represent how many times the destination will be visited before the end of the BFS. 0 is the base case (first visit=stop) ~150, or better, none is the opposite limit. foundLimitAngles is for the complex version of the min path that consider also the different ending sectors.
    static const int nPoints = 50;  // It is the number of points that the function sampleNPoints will sample from the computed vector of vector retrieved from the minPath.    
    constexpr double initialDistAllowed = 20.0; // In case of a starting position of the robot inside the border (not the obstacle) is it allowed to move inside it for a short path, defined in cells size.


	vector<Configuration2<double> > planning(const Mat & img);
	void createMapp();

    vector<Point2<int> > convertToVP(const vector<vector<Point2<int> > > & arr);
    vector<Point2<int> > convertToVP(const vector<vector<Configuration2<double> > > & arr);
    vector<Configuration2<double> > convertToVC(const vector<vector<Configuration2<double> > > & arr);
    vector<Configuration2<double> > convertToVC(const vector<vector<Point2<int> > > & arr);

    void draw(const vector<vector<Point2<int> > > & vv, string name);
    void draw(const vector<vector<Configuration2<double> > > & vv, string name);
    void draw(  const vector<vector<Configuration2<double> > > & vv, 
                const vector<Configuration2<double> > & left, 
                const vector<Configuration2<double> > & right, 
                string name);

	void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn);
	void loadVP(vector<Point2<int> > & vp, FileNode fn);

	int ** allocateAAInt(const int a, const int b);
	int *** allocateAAAInt(const int a, const int b, const int c);
	int **** allocateAAAAInt(const int a, const int b, const int c, const int d);
	double ** allocateAADouble(const int a, const int b);
	double *** allocateAAADouble(const int a, const int b, const int c);
	double **** allocateAAAADouble(const int a, const int b, const int c, const int d);
	Point2<int> ** allocateAAPointInt(const int a, const int b);

    template <class T>
    void deleteAA(T ** arr, const int a){
        for(int i=0; i<a; i++){
            delete[] arr[i];
        }
        delete[] arr;
    }

    template <class T>
    void deleteAAA(T *** arr, const int a, const int b){
        for(int i=0; i<a; i++){
            deleteAA(arr[i], b);
        }
        delete[] arr;
    }

    template <class T>
    void deleteAAAA(T **** arr, const int a, const int b, const int c){
        for(int i=0; i<a; i++){
            deleteAAA(arr[i], b, c);
        }
        delete[] arr;
    }



    vector<vector<Point2<int> > > minPathNPointsWithChoice(const vector<Point2<int> > & vp, const double bonus, const bool angle);
    vector<vector<Point2<int> > > minPathNPoints(const vector<Point2<int> > & vp, const bool angle);
    vector<Point2<int> > minPathTwoPoints(const Point2<int> & p0, const Point2<int> & p1, const bool angle);
	    vector<Point2<int> > minPathTwoPointsInternal(
	                            const Point2<int> & startP, const Point2<int> & endP, 
	                            double ** distances, Point2<int> ** parents);
	    vector<Point2<int> > minPathTwoPointsInternalAngles(
                            const Point2<int> & startP, const Point2<int> & endP, 
                            double *** distances, int **** parents,
                            const double initialDir);

    	int angleSector(const double & d);
    	void intToVect(int c, vector<int> & v);
	    void resetDistanceMap(double ** distances,  const double value = baseDistance);
	    void resetDistanceMap(double *** distances, const double value = baseDistance);
    
    vector<Point2<int> > sampleNPoints(const vector<vector<Point2<int> > > & vvp, const int n=nPoints);
        vector<Point2<int> > sampleNPoints(const vector<Point2<int> > & points, const int n);
        vector<Point2<int> > samplePointsEachNCells(const vector<Point2<int> > & points, const int step);

	void fromVcToPath(vector<Configuration2<double> > & vp, Path & path);
    int getNPoints();

	void plan_dubins(const Configuration2<double>& _start, vector<vector<Configuration2<double> > >& vConfs);
}



#endif