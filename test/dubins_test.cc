#include "../src/dubins.hh"
#include "../src/utils.hh"
#include <cstdio>
#include <ctime>

#define POINTS 3

void printA (Tuple<Point2<double> > arc, string st, FILE* fl){
	fprintf(fl, "%s  <", st.c_str());
  for (int i=0; i<arc.size(); i++) {
    fprintf(fl, "(%f, %f)", arc.get(i).x(), arc.get(i).y());
    if (i!=arc.size()-1){
      fprintf(fl, ", ");
    }
  }
  fprintf(fl, ">\n");
}

int main (){
	int i=0;
	double sum=0.0;
	FILE* fl=fopen("data/test/CC_points.test", "w");
	clock_t begin = clock();
	for (double x0 = 0; x0 <= 150; x0+=25)
  {
    for (double y0 = 0; y0 <= 100; y0+=25)
    {
      for (double th0 = 0; th0 <= 2*M_PI; th0+=0.5)
      {
        for (double x1 = 0; x1 <= 150; x1+=25)
        {
          for (double y1 = 0; y1 <= 100; y1+=25)
          {
            for (double th1 = 0; th1 <= 2*M_PI; th1+=0.5)
            {
							
              Dubins<double> d=Dubins<double>(
								Configuration2<double>(x0, y0, Angle(th0, Angle::RAD)),
								Configuration2<double>(x1, y1, Angle(th1, Angle::RAD))
							);
							int id=d.getId();
							if (id>=0){
		            fprintf(fl, "%f, %f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1, d.length()/POINTS); 
								// fprintf(fl, "%f, %f, %f, %f, %f, %f, %d\n", x0, y0, th0, x1, y1, th1, id<0 ? id : id+1); 
	              Tuple<Tuple<Point2<double> > > ret = d.splitIt(0, d.length()/POINTS);
	              i++;
	              printA(ret.get(0), "arc1", fl);
	              printA(ret.get(1), "arc2", fl);
	              printA(ret.get(2), "arc3", fl);
	            }

							clock_t end = clock();
							double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
							if (elapsed_secs>10){
								printf("%f, %f, %f, %f, %f, %f, media: %fs\n", x0, y0, th0, x1, y1, th1, sum/i);
								begin=clock();
							}
							// fprintf(fl, "\n");
            } 
          } 
        } 
      } 
    } 
  }
	return 0;
}




// #include "/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/src/dubins.hh"
// #include "/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/src/maths.hh"

// #include <iostream>

// using namespace std;

// int main (){
//   Configuration2<double> P0 (0.0, 0.0, Angle(0.0, Angle::RAD));
//   Configuration2<double> P1 (0.0, 25.0, Angle(0.0, Angle::RAD));
//   Dubins<double> d (P0, P1);
//   fprintf(stdout, "a1: %f, %f, %f, %f, %f, %f, %f, %f\n",d.getA1().begin().x(), d.getA1().begin().y(), d.getA1().begin().angle(), 
//   	d.getA1().end().x(), d.getA1().end().y(), d.getA1().end().angle(), d.getA1().length(), d.getA1().getK());
//   fprintf(stdout, "a2: %f, %f, %f, %f, %f, %f, %f, %f\n",d.getA2().begin().x(), d.getA2().begin().y(), d.getA2().begin().angle(), 
//   	d.getA2().end().x(), d.getA2().end().y(), d.getA2().end().angle(), d.getA2().length(), d.getA2().getK());
//   fprintf(stdout, "a3: %f, %f, %f, %f, %f, %f, %f, %f\n",d.getA3().begin().x(), d.getA3().begin().y(), d.getA3().begin().angle(), 
//   	d.getA3().end().x(), d.getA3().end().y(), d.getA3().end().angle(), d.getA3().length(), d.getA3().getK());
//   cout << endl << d << endl;
//   return 0;
// }
