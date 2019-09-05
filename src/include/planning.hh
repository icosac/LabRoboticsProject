#ifndef PLANNING_HH
#define PLANNING_HH

#include <iostream>
#include <tuple>
#include <vector>

#include <map.hh>
#include <utils.hh>
#include <maths.hh>
#include <settings.hh>
#include <objects.hh>
#include <detection.hh>
#include "path.h"

using namespace std;
using namespace cv;

pair< vector<Point2<int> >, Mapp* > planning(const Mat & img);
Mapp * createMapp();

void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn);
void loadVP(vector<Point2<int> > & vp, FileNode fn);

void fromVpToPath(const vector<Point2<int> > & vp, Path & path);

#endif