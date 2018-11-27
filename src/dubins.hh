#ifndef DUBIINS_HH
#define DUBIINS_HH

#include "maths.hh"

#include <iostream>
#include <vector>

using namespace std;

template <class T>
class Curve {
protected:
	Configuration2<T> P0;
	Configuration2<T> P1;

public:
	Curve () : P0(), P1() {}
	Curve (	const Configuration2<T> _P0, 
					const Configuration2<T> _P1) :
			P0(_P0), P1(_P1) {}

	Curve (	const Point2<T> _P0, 
					const Point2<T> _P1, 
					const Angle _th0, 
					const Angle _th1) : 
			P0(_P0, _th0), P1(_P1, _th1) {}

	Curve (	const T x0, const T y0, 
					const Angle _th0,
					const T x1, const T y1, 
					const Angle _th1) :
			P0(x0, y0, _th0), P1(x1, y1, _th1) {}

	Configuration2<T> begin()	const { return P0; }
	Configuration2<T> end()		const { return P1; }

	friend ostream& operator<<(ostream &out, const Curve& data) {
    out << "begin: " << data.begin();
    out << ", end: " << data.end();
    return out;
  }
};


// double Atannn2(double X, double Y){
// 	double arctan=atan(Y/X);
// 	if (X>0)
// 		return arctan;
// 	else if (X<0){
// 		if (Y>=0)
// 			return arctan+M_PI;
// 		else 
// 			return arctan-M_PI;
// 	}
// 	else {
// 		if (Y>0)
// 			return M_PI;
// 		else if (Y<0) 
// 			return 0-M_PI;
// 		else 
// 			return M_PI+1; //INVALID VALUE
// 	}
// }

#define KMAX 1.0
template<class T>
class Dubins : private Curve<T>
{
private:
	double Kmax;

	using Curve<T>::Curve;
public:
	Dubins () : Kmax(KMAX), Curve<T>() {}
	Dubins (const Configuration2<T> _P0, 
					const Configuration2<T> _P1, 
					const double _K=KMAX) :
			Curve<T>(_P0, _P1), Kmax(_K) {}

	Dubins (const Point2<T> _P0, 
					const Point2<T> _P1, 
					const Angle _th0, 
					const Angle _th1, 
					const double _K=KMAX) : 
			Curve<T>(_P0, _P1, _th0, _th1), Kmax(_K) {}

	Dubins (const T x0, const T y0, 
					const Angle _th0,
					const T x1, const T y1, 
					const Angle _th1, 
					const double _K=KMAX) :
			Curve<T>(x0, y0, _th0, x1, y1, _th1), Kmax(_K) {}
	
	double getKMax () const { return Kmax; }

	Tuple<double> LSL (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th1.cos()-th0.cos();
		double S = 2*_kmax + th0.sin()-th1.sin();

		Angle temp1 (atan2(C,S), Angle::RAD);
		Angle sc_s1 = (temp1-th0)*invK;

		double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
		if (temp2<0){
			return Tuple<double>(0);
		}
		double sc_s2 = invK * sqrt(temp2);
		Angle sc_s3 = (th1-temp1)*invK;

		return Tuple<double> (3, sc_s1.get(), sc_s2, sc_s3.get());
	}

	Tuple<double> RSR (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th0.cos()-th1.cos();
		double S = 2*_kmax - th0.sin()+th1.sin();

		Angle temp1 (atan2(C,S), Angle::RAD);
		Angle sc_s1 = (th0-temp1)*invK;

		double temp2 = 2+4*pow2(_kmax) -2*(th0-th1).cos()+4*_kmax*(th0.sin()-th1.sin());
		if (temp2<0){
			return Tuple<double>(0);
		}
		double sc_s2 = invK * sqrt(temp2);
		Angle sc_s3 = (temp1-th1)*invK;

		return Tuple<double> (3, sc_s1.get(), sc_s2, sc_s3.get());
	}

	Tuple<double> LSR (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th0.cos()+th1.cos();
		double S = 2*_kmax + th0.sin()+th1.sin();

		Angle temp1 (atan2(-C,S), Angle::RAD);

		double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax * (th0.sin() + th1.sin());
		if (temp2<0){
			return Tuple<double>(0);
		}
		double sc_s2 = invK * sqrt(temp2);
		Angle temp3 = Angle(-atan2(-2, sc_s2*_kmax), Angle::RAD);
		double sc_s1 = invK * (temp1+temp3-th0).get();
		double sc_s3 = invK * (temp1+temp3-th1).get();

		return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
	}

	Tuple<double> RSL (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th0.cos()+th1.cos();
		double S = 2*_kmax - th0.sin()-th1.sin();

		Angle temp1 (atan2(C,S), Angle::RAD);

		double temp2 = 4*pow2(_kmax) - 2 + 2*(th0-th1).cos() + 4*_kmax*(th0.sin() + th1.sin());
		if (temp2<0){
			return Tuple<double>(0);
		}
		double sc_s2 = invK * sqrt(temp2);
		Angle temp3 = Angle(-atan2(-2, sc_s2*_kmax), Angle::RAD);
		double sc_s1 = ((th0-temp1+temp3)*invK).get();
		double sc_s3 = ((th1-temp1+temp3)*invK).get();

		return Tuple<double> (3, sc_s1, sc_s2, sc_s3);
	}

	Tuple<double> RLR (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th0.cos()-th1.cos();
		double S = 2*_kmax - th0.sin()+th1.sin();

		Angle temp1 (atan2(C,S), Angle::RAD);
		double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() + 4*_kmax*(th0.sin()-th1.sin()));

		if (std::abs(temp2)>1){
			return Tuple<double>(0);
		}
		Angle sc_s2 = Angle(2*M_PI-acos(temp2), Angle::RAD)*invK;
		Angle sc_s1 = (th0-temp1+sc_s2*(0.5*_kmax))*invK;
		Angle sc_s3 = (th1-th0+(sc_s2-sc_s1)*_kmax)*invK;

		return Tuple<double> (3, sc_s1.get(), sc_s2.get(), sc_s3.get());
	}

	Tuple<double> LRL (Angle th0, Angle th1, double _kmax) 
	{
		double invK = 1/_kmax;
		double C = th1.cos()-th0.cos();
		double S = 2*_kmax + th0.sin()-th1.sin();

		Angle temp1 (atan2(C,S), Angle::RAD);
		double temp2 = 0.125*(6 - 4*pow2(_kmax) + 2*(th0-th1).cos() + 4*_kmax*(th0.sin()-th1.sin()));

		if (std::abs(temp2)>1){
			return Tuple<double>(0);
		}
		Angle sc_s2 = Angle(2*M_PI-acos(temp2), Angle::RAD)*invK;
		Angle sc_s1 = (temp1-th0+sc_s2*(0.5*_kmax))*invK;
		Angle sc_s3 = (th1-th0+(sc_s2-sc_s1)*_kmax)*invK;

		return Tuple<double> (3, sc_s1.get(), sc_s2.get(), sc_s3.get());
	}

	Tuple<double> 	scaleToStandard ()
	{
		double dx=Curve<T>::end().x() - Curve<T>::begin().x();
		double dy=Curve<T>::end().y() - Curve<T>::begin().y();

		double _phi=atan2(dy, dx);

		if (_phi!=M_PI+1)
		{

			Angle phi= Angle(_phi, Angle::RAD);
			
			double lambda=sqrt(pow2(dx)+pow2(dy));

			double C = dx /lambda;
			double S = dy /lambda;

			lambda /= 2;

			Angle sc_th0 = Angle (Curve<T>::P0.angle().get()-phi.get(), Angle::RAD);
			Angle sc_th1 = Angle (Curve<T>::P1.angle().get()-phi.get(), Angle::RAD);
			double sc_Kmax = Kmax*lambda;

			return Tuple<double> (4, sc_th0.get(), sc_th1.get(), sc_Kmax, lambda);
		} else 
		{
			return Tuple<double> (0);
		} 
	}

	Tuple<double> scaleFromStandard(double lambda, 
																	Angle sc_s1, 
																	Angle sc_s2, 
																	Angle sc_s3){
		return Tuple<double> (3, (sc_s1 * lambda).get(), 
													(sc_s2 * lambda).get(), 
													(sc_s3 * lambda).get());
	}

	Tuple<double> shortest_path()
	{	
		Tuple<double> scaled = scaleToStandard();
		if (scaled.size()==0)
		{
			return Tuple<double>(0);
		} 
		else 
		{
			Angle 	sc_th0		=	Angle(scaled.get(0), Angle::RAD);
			Angle 	sc_th1		=	Angle(scaled.get(1), Angle::RAD);
			double 	sc_Kmax		=	scaled.get(2);
			double 	sc_lambda	=	scaled.get(3);

			//TODO Missing ksigns matrix
			double Length=0.0;
			bool first_go=true;
			double sc_s1=0.0;
			double sc_s2=0.0;
			double sc_s3=0.0;
			int pidx=-1;

			std::vector<Tuple<double> > res;

			res.push_back(LSL(sc_th0, sc_th1, sc_Kmax));
			res.push_back(RSR(sc_th0, sc_th1, sc_Kmax));
			res.push_back(LSR(sc_th0, sc_th1, sc_Kmax));
			res.push_back(RSL(sc_th0, sc_th1, sc_Kmax));
			res.push_back(RLR(sc_th0, sc_th1, sc_Kmax));
			res.push_back(LRL(sc_th0, sc_th1, sc_Kmax));

			int i=1;
			for (auto value : res){
				if (value.size()>0){
					double appL=value.get(0)+value.get(1)+value.get(2);
					if (appL<Length || first_go){
						first_go=true;
						Length = appL;
						sc_s1=value.get(0);
						sc_s2=value.get(1);
						sc_s3=value.get(2);
						pidx=i;
					}
				}
				i++;
			}	

			if (pidx>0){
				Angle scS1(sc_s1, Angle::RAD);
				Angle scS2(sc_s2, Angle::RAD);
				Angle scS3(sc_s3, Angle::RAD);
				Tuple<double> sc_std = scaleFromStandard(sc_lambda, scS1, scS2, scS3);
			}
		}
		return Tuple<double> (0);
	}

	friend ostream& operator<<(ostream &out, const Dubins& data) {
    out << "begin: " << data.Curve<T>::begin();
    out << ", end: " << data.Curve<T>::end();
    out << ", Kmax: " << data.getKMax();
    return out;
  }

};

#endif