#include "../src/dubins.hh"
#include "../src/utils.hh"
#include <cstdio>
#include <ctime>

int main (){
	// CLEARFILE("data/test/CC_LSL.test");
	// CLEARFILE("data/test/CC_RSR.test");
	// CLEARFILE("data/test/CC_LSR.test");
	// CLEARFILE("data/test/CC_RSL.test");
	// CLEARFILE("data/test/CC_RLR.test");
	// CLEARFILE("data/test/CC_LRL.test");
	// CLEARFILE("data/test/CC_coordinates.test");

	// clock_t begin = clock();
	// for (double x0 = 0; x0 <= 150; x0+=4)
 //  {
 //    for (double y0 = 0; y0 <= 100; y0+=4)
 //    {
 //      for (double th0 = 0; th0 <= 2*M_PI; th0+=0.4)
 //      {
 //        for (double x1 = 0; x1 <= 150; x1+=4)
 //        {
 //          for (double y1 = 0; y1 <= 100; y1+=4)
 //          {
 //            for (double th1 = 0; th1 <= 2*M_PI; th1+=0.4)
 //            {
							
 //              Dubins<double> d=Dubins<double>(
	// 							Configuration2<double>(x0, y0, Angle(th0, Angle::RAD)),
	// 							Configuration2<double>(x1, y1, Angle(th1, Angle::RAD))
	// 						);
	// 						char output[256];
	// 						sprintf(output, "%f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1);
	// 						TOFILE("data/test/CC_coordinates.test", output);
	// 						clock_t end = clock();
	// 						double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	// 						if (elapsed_secs>10){
	// 							printf("%f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1);
	// 							begin=clock();
	// 						}
 //            } 
 //          } 
 //        } 
 //      } 
 //    } 
 //  }

	Dubins<double> d=Dubins<double>(
			Configuration2<double>(0, 0, Angle((-2.0/3.0)*M_PI, Angle::RAD)),
			Configuration2<double>(4, 0, Angle(M_PI/3.0, Angle::RAD))
		);
	cout << d << endl;
	return 0;
}