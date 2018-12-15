#include "../src/maths.hh"
#include "../src/utils.hh"
#include <cmath>



Tuple<double> LSL (Angle th0, Angle th1, double _kmax)
{
  double C=th1.cos()-th0.cos();
  double S=2*_kmax+th0.sin()-th1.sin();
  
  double temp1=2+4*pow2(_kmax)-2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
  
  if (temp1<0){
    return Tuple<double> (0);
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
    return Tuple<double> (0);
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
    return Tuple<double> (0);
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
    return Tuple<double> (0);
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
    return Tuple<double> (0);
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
    return Tuple<double> (0);
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

void printTuple (FILE* fl, Tuple<double> T){
  fprintf(fl, "<");
  for (int i=0; i<T.size(); i++){
    if (i!=T.size()-1){
      fprintf(fl, "%f, ", T.get(i));
    }
    else {
      fprintf(fl, "%f", T.get(i));
    }
  }
  fprintf(fl, ">\n");
}

int main (){
  FILE *LSL_f, *LSR_f, *RSR_f, *RSL_f, *RLR_f, *LRL_f;

  #ifdef DEFAULT
  
  FILE* def=fopen("data/test/dubinstest.test", "w");
  LSL_f=def;
  LSR_f=LSL_f;
  RSR_f=LSL_f;
  RSL_f=LSL_f;
  LRL_f=LSL_f;
  RLR_f=LSL_f;
  
  #else 
  
  // LSL_f=fopen("data/test/CC_LSL.test", "w");
  // LSR_f=fopen("data/test/CC_LSR.test", "w");
  // RSR_f=fopen("data/test/CC_RSR.test", "w");
  // RSL_f=fopen("data/test/CC_RSL.test", "w");
  RLR_f=fopen("data/test/CC_RLR.test", "w");
  LRL_f=fopen("data/test/CC_LRL.test", "w");

  #endif

  for (double th0=0.0; th0<=2*M_PI; th0+=0.1){
    for (double th1=0.0; th1<=2*M_PI; th1+=0.1){
      for (double kmax=0.0; kmax<=5; kmax+=0.1){
        Angle _th0=Angle(th0, Angle::RAD);
        Angle _th1=Angle(th1, Angle::RAD);
        // fprintf(LSL_f, "%f, %f, %f, ", th0, th1, kmax);
        // printTuple(LSL_f, LSL(_th0, _th1, kmax));
        
        // fprintf(LSR_f, "%f, %f, %f, ", th0, th1, kmax);
        // printTuple(LSR_f, LSR(_th0, _th1, kmax));
        
        // fprintf(RSR_f, "%f, %f, %f, ", th0, th1, kmax);
        // printTuple(RSR_f, RSR(_th0, _th1, kmax));
        
        // fprintf(RSL_f, "%f, %f, %f, ", th0, th1, kmax);
        // printTuple(RSL_f, RSL(_th0, _th1, kmax));
        
        fprintf(LRL_f, "%f, %f, %f, ", th0, th1, kmax);
        printTuple(LRL_f, LRL(_th0, _th1, kmax));
        
        fprintf(RLR_f, "%f, %f, %f, ", th0, th1, kmax);
        printTuple(RLR_f, RLR(_th0, _th1, kmax));
      }
    }
  }
  #ifdef DEFAULT

  fclose(def);
  
  #else 

  // fclose(LSL_f);
  // fclose(LSR_f);
  // fclose(RSR_f);
  // fclose(RSL_f);
  fclose(RLR_f);
  fclose(LRL_f);

  #endif

  return 0;
}
