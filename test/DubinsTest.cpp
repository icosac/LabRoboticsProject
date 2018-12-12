#include "../src/maths.hh"
#include "../src/utils.hh"
#include <cmath>

#define FORMULA

Tuple<double> LSL (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C=th0.cos()-th1.cos();
  double S=2*_kmax-th0.sin()+th1.sin();
  
  double temp1=2+4*pow2(_kmax)-(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
  
  if (temp1<0){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  Angle sc_s1=Angle(th0.get()-atan2(C,S), Angle::RAD)*invK;
  double sc_s2=invK*sqrt(temp1);
  Angle sc_s3=Angle(atan2(C,S)-th1.get(), Angle::RAD)*invK;
  
  return Tuple<double> (3, sc_s1.get(), sc_s2, sc_s3.get());
  
#else
  
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
#endif
}

Tuple<double> RSR (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C=th0.cos()-th1.cos();
  double S=2*_kmax-th0.sin()+th1.sin();
  
  double temp1=2+4*pow2(_kmax)-(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
  
  if (temp1<0){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  Angle sc_s1=Angle(th0.get()-atan2(C,S), Angle::RAD)*invK;
  double sc_s2=invK*sqrt(temp1);
  Angle sc_s3=Angle(atan2(C,S)-th1.get(), Angle::RAD)*invK;
  
  return Tuple<double> (3, sc_s1.get(), sc_s2, sc_s3.get());
#else
  
  double invK = 1/_kmax;
  double C = th0.cos()-th1.cos();
  double S = 2*_kmax - th0.sin()+th1.sin();
  
  Angle temp1 (atan2(C,S), Angle::RAD);
  double sc_s1 = invK*(th0-temp1).get();
  
  double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
  if (temp2<0){
    return Tuple<double>(0);
  }
  double sc_s2 = invK * sqrt(temp2);
  double sc_s3 = invK*(temp1-th1).get();
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
#endif
}

Tuple<double> LSR (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C = th0.cos()+th1.cos();
  double S=2*_kmax+th0.sin()+th1.sin();
  
  double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()+th1.sin());
  if (temp1<0){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  
  double sc_s2=invK*sqrt(temp1);
  Angle sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0.get(), Angle::RAD)*invK;
  Angle sc_s3= Angle(atan2(-C,S)-atan2(2, _kmax*sc_s2)-th1.get(), Angle::RAD)*invK;
  
  return Tuple<double>(3, sc_s1.get(), sc_s2, sc_s3.get());
#else
  double invK = 1/_kmax;
  double C = th0.cos()+th1.cos();
  double S = 2*_kmax + th0.sin()+th1.sin();
  
  double temp1=atan2(-C,S);
  
  double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax * (th0.sin() + th1.sin());
  if (temp2<0){
    return Tuple<double>(0);
  }
  double sc_s2 = invK * sqrt(temp2);
  double temp3 = -atan2(-2, sc_s2*_kmax);
  double sc_s1 = invK * Angle((temp3-th0.get()+temp1), Angle::RAD).get();
  double sc_s3 = invK * Angle((temp3-th1.get()+temp1), Angle::RAD).get();
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
#endif
}

Tuple<double> RSL (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C = th0.cos()+th1.cos();
  double S=2*_kmax-th0.sin()-th1.sin();
  
  double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()+th1.sin());
  if (temp1<0){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  
  double sc_s2=invK*sqrt(temp1);
  Angle sc_s1= Angle(th0.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD)*invK;
  Angle sc_s3= Angle(th1.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD)*invK;
  
  return Tuple<double>(3, sc_s1.get(), sc_s2, sc_s3.get());
#else
  double invK = 1/_kmax;
  double C = th0.cos()+th1.cos();
  double S = 2*_kmax - th0.sin()-th1.sin();
  
  double temp1=atan2(C,S);
  
  double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax*(th0.sin() + th1.sin());
  if (temp2<0){
    return Tuple<double>(0);
  }
  double sc_s2 = invK * sqrt(temp2);
  double temp3 = -atan2(-2, sc_s2*_kmax);
  double sc_s1 = invK * Angle((th0.get()-temp1+temp3), Angle::RAD).get();
  double sc_s3 = invK * Angle((th1.get()-temp1+temp3), Angle::RAD).get();
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
#endif
}

Tuple<double> RLR (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C=th0.cos()-th1.cos();
  double S=2*_kmax-th0.sin()+th1.sin();
  
  double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin()));
  
  if (std::abs(temp1)>1){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  Angle sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD)*invK;
  Angle sc_s1 = Angle(th0.get()-atan2(C, S)+0.5*_kmax*sc_s2.get(), Angle::RAD)*invK;
  Angle sc_s3 = Angle(th0.get()-th1.get()+_kmax*(sc_s2-sc_s1).get(), Angle::RAD)*invK;
  
  return Tuple<double>(3, sc_s1.get(), sc_s2.get(), sc_s3.get());
  
#else
  double invK = 1/_kmax;
  double C = th0.cos()-th1.cos();
  double S = 2*_kmax - th0.sin()+th1.sin();
  
  Angle temp1 (atan2(C,S), Angle::RAD);
  double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() + 4*_kmax*(th0.sin()-th1.sin()));
  if (std::abs(temp2)>1){
    return Tuple<double>(0);
  }
  
  double sc_s2 = invK*(Angle(2*M_PI-acos(temp2), Angle::RAD).get());
  double sc_s1 = invK*(th0-temp1+Angle(sc_s2*(0.5*_kmax), Angle::RAD)).get();
  double sc_s3 = invK*Angle(th1.get()-th0.get()+(sc_s2-sc_s1)*_kmax, Angle::RAD).get();
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
#endif
}

Tuple<double> LRL (Angle th0, Angle th1, double _kmax)
{
#if 1
  double C=th1.cos()-th0.cos();
  double S=2*_kmax+th0.sin()-th1.sin();
  
  double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin()));
  
  if (std::abs(temp1)>1){
    return Tuple<double> (0);
  }
  
  double invK=1/_kmax;
  Angle sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD)*invK;
  Angle sc_s1 = Angle(atan2(C, S)-th0.get()+0.5*_kmax*sc_s2.get(), Angle::RAD)*invK;
  Angle sc_s3 = Angle(th1.get()-th0.get()+_kmax*(sc_s2-sc_s1).get(), Angle::RAD)*invK;
  
  return Tuple<double>(3, sc_s1.get(), sc_s2.get(), sc_s3.get());
#else
  double invK = 1/_kmax;
  double C = th1.cos()-th0.cos();
  double S = 2*_kmax + th0.sin()-th1.sin();
  
  Angle temp1 (atan2(C,S), Angle::RAD);
  double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() - 4*_kmax*(th0.sin()-th1.sin()));
  
  if (std::abs(temp2)>1){
    return Tuple<double>(0);
  }
  double sc_s2 = invK*(Angle(2*M_PI-acos(temp2), Angle::RAD)).get();
  double sc_s1 = invK*(temp1-th0+Angle(sc_s2*(0.5*_kmax), Angle::RAD)).get();
  double sc_s3 = invK*Angle(th1.get()-th0.get()+(sc_s2-sc_s1)*_kmax, Angle::RAD).get();
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
#endif
}

void printTuple (Tuple<double> T){
  printf("<");
  for (int i=0; i<T.size(); i++){
    if (i!=T.size()-1){
      printf("%f, ", T.get(i));
    }
    else {
      printf("%f", T.get(i));
    }
  }
  printf(">\n");
}

int main (){
  for (double th0=0.0; th0<=2*M_PI; th0+=0.1){
    for (double th1=0.0; th1<=2*M_PI; th1+=0.1){
      for (double kmax=0.0; kmax<=5; kmax+=0.1){
        Angle _th0=Angle(th0, Angle::RAD);
        Angle _th1=Angle(th1, Angle::RAD);
        printf("%f, %f, %f, ", th0, th1, kmax);
        printTuple(LSL(_th0, _th1, kmax));
        
        // printf("%f, %f, %f, ", th0, th1, kmax);
        // printTuple(LSR(_th0, _th1, kmax));
        
        // printf("%f, %f, %f, ", th0, th1, kmax);
        // printTuple(RSR(_th0, _th1, kmax));
        
        // printf("%f, %f, %f, ", th0, th1, kmax);
        // printTuple(RSL(_th0, _th1, kmax));
        
        // printf("%f, %f, %f, ", th0, th1, kmax);
        // printTuple(LRL(_th0, _th1, kmax));
        
        // printf("%f, %f, %f, ", th0, th1, kmax);
        // printTuple(RLR(_th0, _th1, kmax));
        
        // printf("\n");
      }
    }
  }
  return 0;
}
