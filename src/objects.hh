#ifndef OBJECTS
#define OBJECTS

#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "../clipper/clipper.hpp"

using namespace cv;
using namespace std;

class Object{
    public:
        string toString();
        void print();

        unsigned size();
        unsigned nPoint();

        void computeCenter();
        void computeRadius();
        
        void offsetting(int offset);
        bool collideApproximate(Point p);
        virtual bool collide(Point p) = 0;

    protected:
        vector<Point> points;
        Point center;
        float radius;
};

class Obstacle: public Object{
    public:
    Obstacle(vector<Point> vp);
    string toString();
    void print();

    //void offsetting();
    bool collide(Point p){
        cout << "The function 'collide' now return always true\n";
        return(true);
    }
};


class Victim: public Object{
public:
    Victim(vector<Point> vp, int _value);
    string toString();
    void print();

    //void offsetting();
    bool collide(Point p){
        cout << "The function 'collide' now return always true\n";
        return(true);
    }

    int getValue()
        {return(value);}
    void setValue(int v)
        {value=v;}

protected:
    int value;
};

#endif

//compile command:
//g++ `pkg-config --cflags opencv` -std=c++11 -Wall -O3  -o objects.out objects.cc `pkg-config --libs opencv`