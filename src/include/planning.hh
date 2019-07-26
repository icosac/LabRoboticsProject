#ifndef PLANNING_HH
#define PLANNING_HH

#include <iostream>
#include <tuple>
#include <vector>

#include <map.hh>
#include <utils.hh>
#include <maths.hh>
#include <settings.hh>

using namespace std;
using namespace cv;

pair< vector<Point2<int> >, Mapp* > planning();
Mapp * createMapp();

void loadVVP(vector<vector<Point2<int> > > & vvp, FileNode fn);
void loadVP(vector<Point2<int> > & vp, FileNode fn);

#endif