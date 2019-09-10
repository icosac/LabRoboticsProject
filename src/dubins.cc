#include "dubins.hh"

Configuration2<double> circline(double _L,
                                Configuration2<double> _P0,
                                double _K)
{
  double sincc=_L*sinc(_K*_L/2.0);
  double app=_K*_L/2.0;
  double phi=_P0.angle().toRad();
  
  double x=_P0.x() + sincc * cos(phi+app);
  double y=_P0.y() + sincc * sin(phi+app);
  Angle th=Angle(_K*_L+phi, Angle::RAD);

  return Configuration2<double>(x, y, th.get());
}

Tuple<Angle> toBase(Tuple<Angle> z, int n, int base, const Angle& inc, int startPos, int endPos){
  int i=z.size()-1;
  do {
    if (i<startPos || i>endPos){}
    else {
      z.set(i, (z.get(i)+Angle(inc.toRad()*(n%base), Angle::RAD)));
      n=(int)(n/base);
    }
    i--;
  } while(n!=0 && i>-1);

  return z;
}

void disp(Tuple<Tuple<Angle> >& t,
          Tuple<Angle>& z,    //Vector to use
          int N,              //Number of time to "iterate"
          const Angle& inc,   //Incrementation
          int startPos,			//=0 		
          int endPos){ 			//=0		
  unsigned long M=z.size()-startPos;
  if (endPos!=0 && endPos>startPos){
    M-=(z.size()-endPos-1);
  }
  unsigned long iter_n=pow(N, M);
  for (unsigned long i=0; i<iter_n; i++){
      Tuple<Angle> app=toBase(z, i, N, inc, startPos, endPos);
      t.add(app);
  }
  #ifdef DEBUG
  cout << "Expected: " << iter_n << " got: " << t.size() << endl;
  #endif
}