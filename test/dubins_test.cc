#include<dubins.hh>

using namespace std;

#define ALl
#ifdef ALL
int main(){
	Configuration2<double> C0(100.0, 100.0, Angle(315, Angle::DEG));
	Configuration2<double> C1(300.0, 300.0, Angle(251.565, Angle::DEG));

	Dubins<double> d(C0, C1, 0.01);

	cout << "id: " << d.getId() << endl;

	cout << d << endl;

	return 0;
}

#else

#include <chrono>
using namespace chrono;

typedef chrono::high_resolution_clock Clock;

extern Angle A_DEG_NULL;
extern Angle A_360;

#define Kmax 1

Tuple<double> LRL (Angle th0, Angle th1, double _kmax);
Tuple<double> LSL (Angle th0, Angle th1, double _kmax);
Tuple<double> RSL (Angle th0, Angle th1, double _kmax);
Tuple<double> LSR (Angle th0, Angle th1, double _kmax);
Tuple<double> RSR (Angle th0, Angle th1, double _kmax);
Tuple<double> RLR (Angle th0, Angle th1, double _kmax);

Tuple<double> scaleToStandard (double x0, double y0, Angle a0, double x1, double y1, Angle a1);

int main (){
	uint i=0;
	auto gen_start=Clock::now();
	double elapsed=0.0;
	double elapsedAll=0.0;
	auto start=Clock::now();

	for (double x0=0; x0<1000; x0+=10){
		for (double y0=0; y0<1500; y0+=10){
			for (int i=0; i<360; i+=10){
				for (double x1=0; x1<1000; x1+=10){
					for (double y1=0; y1<1500; y1+=10){
						for (int j=0; j<360; j+=10){
							Angle a0 (i, Angle::DEG);
							Angle a1 (j, Angle::DEG);

							start=Clock::now();
							Tuple<double> ret=scaleToStandard(x0, y0, a0, x1, y1, a1);
							double _kmax=ret.get(2);
							Angle th0=Angle(ret.get(0), Angle::RAD);
							Angle th1=Angle(ret.get(1), Angle::RAD);
							Tuple<double> LRL (th0, th1, _kmax);
							Tuple<double> LSL (th0, th1, _kmax);
							Tuple<double> RSL (th0, th1, _kmax);
							Tuple<double> LSR (th0, th1, _kmax);
							Tuple<double> RSR (th0, th1, _kmax);
							Tuple<double> RLR (th0, th1, _kmax);
							
							auto stop=Clock::now();
							double app=duration_cast<nanoseconds>(stop - start).count()/1000.0;
							elapsed+=app;

							start=Clock::now();
							
							Dubins<double> d(x0, y0, a0, x1, y1, a1, Kmax);
							
							stop=Clock::now();
							app=duration_cast<nanoseconds>(stop - start).count()/1000.0;
							elapsedAll+=app;

							if (duration_cast<seconds>(stop - gen_start).count() > 10){
								cout << x0 << " " << y0 << " " << a0 << " " << x1 << " " << y1 << " " << a1 << endl; 
								cout << "ForEach: " << elapsed << " ms" << endl;
								cout << "Dubins: " << elapsedAll << " ms" << endl;
								cout << i << endl;
								gen_start=Clock::now();
							}
							i++;
						}
					}
				}
			}
		}
	}
	cout << "ForEach: " << elapsed << " ms" << endl;
	cout << "Dubins: " << elapsedAll << " ms" << endl;
	cout << i << endl;
	return 0;
}


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

Tuple<double> scaleToStandard (double x0, double y0, Angle a0, double x1, double y1, Angle a1)
{
  double dx=x1 - x0;
  double dy=y1 - y0;

  double _phi=atan2(dy, dx);

  Angle phi= Angle(_phi, Angle::RAD);

  double lambda=sqrt(pow2(dx)+pow2(dy))/2; //hypt

  Angle sc_th0 = a0-phi;
  Angle sc_th1 = a1-phi;

  double sc_Kmax = Kmax*lambda;

  return Tuple<double> (4, sc_th0.toRad(), sc_th1.toRad(), sc_Kmax, lambda);
}

#endif