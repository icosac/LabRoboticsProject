#include "maths.hh"

__kernel void Angles (__global Angle th0, 
                      __global Angle th1){
  printf("th0: %d\n", th0.get());
  printf("th1: %d\n", th1.get());
}

// __kernel void LSL ( __global Angle th0, 
//                     __global Angle th1, 
//                     __global double _kmax, 
//                     __global Tuple<double>* ret)
// {
//   double C=th1.cos()-th0.cos();
//   double S=2*_kmax+th0.sin()-th1.sin();
  
//   double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
//   printf("ok\n");
//   if (temp1<0){
//     printf("Errore\n");
//     ret=new Tuple<double> (0);
//   }
  
//   double invK=1/_kmax;
//   double sc_s1=Angle(atan2(C,S)-th0.get(), Angle::RAD).get()*invK;
//   double sc_s2=invK*sqrt(temp1);
//   double sc_s3=Angle(th1.get()-atan2(C,S), Angle::RAD).get()*invK;

//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }

// __kernel void RSR (__global Angle th0, __global Angle th1, __global double _kmax, __global Tuple<double>* ret)
// {
//   double C=th0.cos()-th1.cos();
//   double S=2*_kmax-th0.sin()+th1.sin();
  
//   double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
  
//   if (temp1<0){
//     ret=new Tuple<double> (0);
//   }
  
//   double invK=1/_kmax;
//   double sc_s1=Angle(th0.get()-atan2(C,S), Angle::RAD).get()*invK;
//   double sc_s2=invK*sqrt(temp1);
//   double sc_s3=Angle(atan2(C,S)-th1.get(), Angle::RAD).get()*invK;
  
//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }

// __kernel void LSR (__global Angle th0, __global Angle th1, __global double _kmax, __global Tuple<double>* ret)
// {
//   double C = th0.cos()+th1.cos();
//   double S=2*_kmax+th0.sin()+th1.sin();
  
//   double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()+th1.sin());
//   if (temp1<0){
//     ret=new Tuple<double> (0);
//   }
  
//   double invK=1/_kmax;
  
//   double sc_s2=invK*sqrt(temp1);
//   double sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0.get(), Angle::RAD).get()*invK;
//   double sc_s3= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1.get(), Angle::RAD).get()*invK;
  
//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }

// __kernel void RSL (__global Angle th0, __global Angle th1, __global double _kmax, __global Tuple<double>* ret)
// {
//   double C = th0.cos()+th1.cos();
//   double S=2*_kmax-th0.sin()-th1.sin();
  
//   double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()+th1.sin());
//   if (temp1<0){
//     ret=new Tuple<double> (0);
//   }
  
//   double invK=1/_kmax;
  
//   double sc_s2=invK*sqrt(temp1);
//   double sc_s1= Angle(th0.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
//   double sc_s3= Angle(th1.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
  
//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }

// __kernel void RLR (__global Angle th0, __global Angle th1, __global double _kmax, __global Tuple<double>* ret)
// {
//   double C=th0.cos()-th1.cos();
//   double S=2*_kmax-th0.sin()+th1.sin();
  
//   double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin()));
  
//   if (fabs(temp1)-Epsi>1.0){
//     ret=new Tuple<double> (0);
//   }

//   if (equal(fabs(temp1), 1.0) ){
//     temp1=round(temp1);
//   }
  
//   double invK=1/_kmax;
//   double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
//   double sc_s1 = Angle(th0.get()-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
//   double sc_s3 = Angle(th0.get()-th1.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
  
//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }

// __kernel void LRL (__global Angle th0, __global Angle th1, __global double _kmax, __global Tuple<double>* ret)
// {
//   double C=th1.cos()-th0.cos();
//   double S=2*_kmax+th0.sin()-th1.sin();
  
//   double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin()));

//   if (fabs(temp1)-Epsi>1.0){
//     ret=new Tuple<double> (0);
//   }

//   if (equal(fabs(temp1), 1.0) ){
//     temp1=round(temp1);
//   }

//   double invK=1/_kmax;
//   double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
//   double sc_s1 = Angle(atan2(C, S)-th0.get()+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
//   double sc_s3 = Angle(th1.get()-th0.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
  
//   ret=new Tuple<double>(3, sc_s1, sc_s2, sc_s3);
// }