#include "maths.hh"
#include "utils.hh"
#include "dubins.hh"
#include <chrono>

Tuple<double> LSL (Angle th0, Angle th1, double _kmax)
{
  double C=th1.cos()-th0.cos();
  double S=2*_kmax+th0.sin()-th1.sin();
  
  double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
  
  if (temp1<0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }
  
  double invK=1/_kmax;
  double sc_s1=Angle(atan2(C,S)-th0.get(), Angle::RAD).get()*invK;
  double sc_s2=invK*sqrt(temp1);
  double sc_s3=Angle(th1.get()-atan2(C,S), Angle::RAD).get()*invK;
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
}

Tuple<double> RSR (Angle th0, Angle th1, double _kmax)
{
  double C=th0.cos()-th1.cos();
  double S=2*_kmax-th0.sin()+th1.sin();
  
  double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin());
  
  if (temp1<0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }
  
  double invK=1/_kmax;
  double sc_s1=Angle(th0.get()-atan2(C,S), Angle::RAD).get()*invK;
  double sc_s2=invK*sqrt(temp1);
  double sc_s3=Angle(atan2(C,S)-th1.get(), Angle::RAD).get()*invK;
  
  return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
}

Tuple<double> LSR (Angle th0, Angle th1, double _kmax)
{
  double C = th0.cos()+th1.cos();
  double S=2*_kmax+th0.sin()+th1.sin();
  
  double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()+th1.sin());
  if (temp1<0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }
  
  double invK=1/_kmax;
  
  double sc_s2=invK*sqrt(temp1);
  double sc_s1= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th0.get(), Angle::RAD).get()*invK;
  double sc_s3= Angle(atan2(-C,S)-atan2(-2, _kmax*sc_s2)-th1.get(), Angle::RAD).get()*invK;
  
  return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

Tuple<double> RSL (Angle th0, Angle th1, double _kmax)
{
  double C = th0.cos()+th1.cos();
  double S=2*_kmax-th0.sin()-th1.sin();
  
  double temp1=-2+4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()+th1.sin());
  if (temp1<0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }
  
  double invK=1/_kmax;
  
  double sc_s2=invK*sqrt(temp1);
  double sc_s1= Angle(th0.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
  double sc_s3= Angle(th1.get()-atan2(C,S)+atan2(2, _kmax*sc_s2), Angle::RAD).get()*invK;
  
  return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

Tuple<double> RLR (Angle th0, Angle th1, double _kmax)
{
  double C=th0.cos()-th1.cos();
  double S=2*_kmax-th0.sin()+th1.sin();
  
  double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin()));
  
  if (fabs(temp1)-Epsi>1.0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }

  if (equal(fabs(temp1), 1.0) ){
    temp1=round(temp1);
  }
  
  double invK=1/_kmax;
  double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
  double sc_s1 = Angle(th0.get()-atan2(C, S)+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
  double sc_s3 = Angle(th0.get()-th1.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
  
  return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

Tuple<double> LRL (Angle th0, Angle th1, double _kmax)
{
  double C=th1.cos()-th0.cos();
  double S=2*_kmax+th0.sin()-th1.sin();
  
  double temp1=0.125*(6-4*pow2(_kmax)+2*(th0-th1).cos()-4*_kmax*(th0.sin()-th1.sin()));

  if (fabs(temp1)-Epsi>1.0){
    return Tuple<double> (3, -1.0, -1.0, -1.0);
  }

  if (equal(fabs(temp1), 1.0) ){
    temp1=round(temp1);
  }

  double invK=1/_kmax;
  double sc_s2 = Angle(2*M_PI-acos(temp1), Angle::RAD).get()*invK;
  double sc_s1 = Angle(atan2(C, S)-th0.get()+0.5*_kmax*sc_s2, Angle::RAD).get()*invK;
  double sc_s3 = Angle(th1.get()-th0.get()+_kmax*(sc_s2-sc_s1), Angle::RAD).get()*invK;
  
  return Tuple<double>(3, sc_s1, sc_s2, sc_s3);
}

typedef std::chrono::high_resolution_clock Clock;

int main (){
  double sum=0.0, avrg=0.0, min=100.0, max=0.0;
  double sum_LSL=0.0, avrg_LSL=0.0, min_LSL=100.0, max_LSL=0.0;
  double sum_RSR=0.0, avrg_RSR=0.0, min_RSR=100.0, max_RSR=0.0;
  double sum_LSR=0.0, avrg_LSR=0.0, min_LSR=100.0, max_LSR=0.0;
  double sum_RSL=0.0, avrg_RSL=0.0, min_RSL=100.0, max_RSL=0.0;
  double sum_LRL=0.0, avrg_LRL=0.0, min_LRL=100.0, max_LRL=0.0;
  double sum_RLR=0.0, avrg_RLR=0.0, min_RLR=100.0, max_RLR=0.0;

  FILE* LSL_f=fopen("data/test/CL/CC/LSL.test", "w");
  FILE* LSR_f=fopen("data/test/CL/CC/LSR.test", "w");
  FILE* RSR_f=fopen("data/test/CL/CC/RSR.test", "w");
  FILE* RSL_f=fopen("data/test/CL/CC/RSL.test", "w");
  FILE* RLR_f=fopen("data/test/CL/CC/RLR.test", "w");
  FILE* LRL_f=fopen("data/test/CL/CC/LRL.test", "w");
  FILE* time_f=fopen("data/test/CL/CC/time.test", "a");

  fprintf(LSL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RSR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(LSR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RSL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(LRL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RLR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");

  int i=0;
  for (double th0=0.0; th0<=2*M_PI; th0+=0.05){
    for (double th1=0.0; th1<=2*M_PI; th1+=0.05){
      for (double kmax=0.0; kmax<=5; kmax+=0.05){
        Angle _th0=Angle(th0, Angle::RAD);
        Angle _th1=Angle(th1, Angle::RAD);

        auto t1=Clock::now();
        Tuple<double> LSL_ret=LSL(_th0, _th1, kmax);
        Tuple<double> RSR_ret=RSR(_th0, _th1, kmax);
        Tuple<double> LSR_ret=LSR(_th0, _th1, kmax);
        Tuple<double> RSL_ret=RSL(_th0, _th1, kmax);
        Tuple<double> LRL_ret=LRL(_th0, _th1, kmax);
        Tuple<double> RLR_ret=RLR(_th0, _th1, kmax);
        
        auto t2=Clock::now();
        double diff=(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count())/1000000.0;
        i++;
        sum+=diff;
        max=(max>diff ? max : diff);
        min=(min<diff ? min : diff);

        fprintf(LSL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, LSL_ret.get(0), LSL_ret.get(1), LSL_ret.get(2));
        fprintf(RSR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, RSR_ret.get(0), RSR_ret.get(1), RSR_ret.get(2));
        fprintf(LSR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, LSR_ret.get(0), LSR_ret.get(1), LSR_ret.get(2));
        fprintf(RSL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, RSL_ret.get(0), RSL_ret.get(1), RSL_ret.get(2));
        fprintf(LRL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, LRL_ret.get(0), LRL_ret.get(1), LRL_ret.get(2));
        fprintf(RLR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, RLR_ret.get(0), RLR_ret.get(1), RLR_ret.get(2));
        
      }
    }
  }
  avrg=sum/i;
  fprintf(time_f, "LSL+RSR+LSR+RSL+LRL+RLR\nTot: %fms, avrg: %fms, min: %fms, max: %fms\n\n", sum, avrg, min, max);
  
  fclose(LSL_f);
  fclose(LSR_f);
  fclose(RSR_f);
  fclose(RSL_f);
  fclose(RLR_f);
  fclose(LRL_f);

  return 0;
}
