#include "../src/maths.hh"
#include "../src/dubins.hh"
#include <cstdio>

#define Kmax 1
typedef double T;

Tuple<double> scaleToStandard (Curve<T> curve)
{
  double dx=curve.end().x() - curve.begin().x();
  double dy=curve.end().y() - curve.begin().y();

  double _phi=atan2(dy, dx);
  Angle phi= Angle(_phi, Angle::RAD);
  double lambda=sqrt(pow2(dx)+pow2(dy));

  double C = dx /lambda;
  double S = dy /lambda;

  lambda /= 2;

  Angle sc_th0 = Angle (curve.begin().angle().get()-phi.get(), Angle::RAD);
  Angle sc_th1 = Angle (curve.end().angle().get()-phi.get(), Angle::RAD);
  double sc_Kmax = Kmax*lambda;

  return Tuple<double> (4, sc_th0.get(), sc_th1.get(), sc_Kmax, lambda);
}

void LSL (Angle th0, Angle th1, double _kmax)
{
  double invK = 1/_kmax;
  double C = th1.cos()-th0.cos();
  double S = 2*_kmax + th0.sin()-th1.sin();

  Angle temp1 (atan2(C,S), Angle::RAD);
  double sc_s1 = invK*(temp1-th0).get();

  double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
  if (temp2<0){
    return Tuple<double>(0);
  }
  double sc_s2 = invK * sqrt(temp2);
  double sc_s3 = invK*(th1-temp1).get();

  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
}


int main (){
  FILE* fl=fopen("../data/test/CC_scale.test", "w");

  for (double x0 = 0; x0 <= 150; x0+=2)
  {
    for (double y0 = 0; y0 <= 100; y0+=2)
    {
      for (double th0 = 0; th0 <= 2*M_PI; th0+=0.2)
      {
        for (double x1 = 0; x1 <= 150; x1+=2)
        {
          for (double y1 = 0; y1 <= 100; y1+=2)
          {
            for (double th1 = 0; th1 <= 2*M_PI; th1+=0.2)
            {
              Curve <double> c (Configuration2<double> (x0, y0, Angle(th0, Angle::RAD)), 
                                Configuration2<double> (x1, y1, Angle(th1, Angle::RAD)));
              Tuple<double> A=scaleToStandard (c);
              fprintf(fl, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1, A.get(0), A.get(1), A.get(2), A.get(3));
              printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1, A.get(0), A.get(1), A.get(2), A.get(3));
            } 
          } 
        } 
      } 
    } 
  }

  fclose(fl);

  return 0;
}
