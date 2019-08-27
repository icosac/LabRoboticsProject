#include "dubins.hh"

double* dubinsSetCuda(Configuration2<double> start,
                      Configuration2<double> end,
                      Tuple<Point2<double> > _points,
                      double _kmax,//=1
                      int startPos,//=0
                      int endPos,//=-1
                      uint parts){//=2
  #define ULONG_SIZE sizeof(unsigned long)*8
  #define ULONG_SIZE2 sizeof(unsigned long)*4
  if (endPos==-1){
    endPos=_points.size();
  }

  if (_points.size()+2<ULONG_SIZE/parts && parts!=2){}
  else if (_points.size()+2<ULONG_SIZE/4){
    parts=16;
  }
  else if (_points.size()+2<ULONG_SIZE/3){
    parts=8;
  }
  else if (_points.size()+2<ULONG_SIZE/2){
    parts=4;
  }
  else if (_points.size()+2<ULONG_SIZE){
    parts=2;
  }
  
  else {
    cerr << "Too many points." << endl;
    return nullptr; 
  }
  
  //Compute number of iteration. Since I need to check `parts` incrementation for each angle it'll be `parts`^(endPos-startPos)
  size_t size=_points.size()+2;
  unsigned long M=size-startPos;
  if (endPos>startPos){
    M-=(size-endPos-1);
  }
  ulong iter_n=pow(parts, M);
  { COUT(parts)
    COUT(size)
    COUT(M)
    COUT(parts)
    COUT(iter_n)}

  double inc=(A_2PI/parts).toRad();

  double* angles=(double*) malloc(sizeof(double)*_points.size()+2);

  int i=0; 
  while (i<1 && inc>0.0001){
    angles=My_CUDA::dubinsSetBest(start, end, _points, startPos, endPos, iter_n, inc, parts, _kmax);
    inc/=parts;
    i++;

    Tuple<Configuration2<double> > confs;
    cout << start.angle().toRad() << " ";
    confs.add(start);
    for (int i=1; i<_points.size()+1; i++){
      cout << angles[i] << " ";
      confs.add(Configuration2<double> (_points.get(i-1), Angle(angles[i], Angle::RAD)));
    }
    cout << end.angle().toRad() << endl;
    confs.add(end);

    DubinsSet<double> s_CUDA (confs, _kmax);

    uint DIMY=750;
    uint DIMX=500;
    Mat map(DIMY, DIMX, CV_8UC3, Scalar(255, 255, 255));
    s_CUDA.draw(DIMX, DIMY, map, 250, 2.5);
    my_imshow("CUDA", map, true);
    mywaitkey();
  }
  
  return angles;
}

/* NOTHING TO SEE HERE MAYBE*/  
