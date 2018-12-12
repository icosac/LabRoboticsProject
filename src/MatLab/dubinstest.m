%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Demo usage of Dubins shortest path function;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reset environment
close all; clear all; clc;

fl= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/DubinsTest_matlab.test", "w");

% value=0;
for th0=0.0 : 0.1 : 2*pi
  for th1=0.0 : 0.1 : 2*pi
    for kmax=0.0 : 0.1 : 5
      [ok, sc_s1, sc_s2, sc_s3]=LSL(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end

      [ok, sc_s1, sc_s2, sc_s3]=LSR(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end
      
      [ok, sc_s1, sc_s2, sc_s3]=RSR(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end
      
      [ok, sc_s1, sc_s2, sc_s3]=RSL(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end
      
      [ok, sc_s1, sc_s2, sc_s3]=LRL(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end

      [ok, sc_s1, sc_s2, sc_s3]=RLR(th0, th1, kmax);
      if ok==true
        fprintf(fl, "%f, %f, %f, <%f, %f, %f>\n", th0, th1, kmax, sc_s1, sc_s2, sc_s3);
      else
        fprintf(fl, "<>\n");
      end
      fprintf(fl, "\n");
    end
  end
end

% Normalize an angle (in range [0,2*pi))
function out = mod2pi(ang);
  out = ang;
  while (out < 0)
    out = out + 2 * pi;
  end
  while (out >= 2 * pi)
    out = out - 2 * pi;
  end
end

% LSL
function [ok, sc_s1, sc_s2, sc_s3] = LSL(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_thf) - cos(sc_th0);
  S = 2 * sc_Kmax + sin(sc_th0) - sin(sc_thf);
  temp1 = atan2(C, S);
  sc_s1 = invK * mod2pi(temp1 - sc_th0);
  temp2 = 2 + 4 * sc_Kmax^2 - 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf));
  if (temp2 < 0)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * sqrt(temp2);
  sc_s3 = invK * mod2pi(sc_thf - temp1);
  ok = true;
end

% RSR
function [ok, sc_s1, sc_s2, sc_s3] = RSR(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_th0) - cos(sc_thf);
  S = 2 * sc_Kmax - sin(sc_th0) + sin(sc_thf);
  temp1 = atan2(C, S);
  sc_s1 = invK * mod2pi(sc_th0 - temp1);
  temp2 = 2 + 4 * sc_Kmax^2 - 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf));
  if (temp2 < 0)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * sqrt(temp2);
  sc_s3 = invK * mod2pi(temp1 - sc_thf);
  ok = true;
end

% LSR
function [ok, sc_s1, sc_s2, sc_s3] = LSR(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_th0) + cos(sc_thf);
  S = 2 * sc_Kmax + sin(sc_th0) + sin(sc_thf);
  temp1 = atan2(-C, S);
  temp3 = 4 * sc_Kmax^2 - 2 + 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) + sin(sc_thf));
  if (temp3 < 0)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * sqrt(temp3);
  temp2 = -atan2(-2, sc_s2 * sc_Kmax);
  sc_s1 = invK * mod2pi(temp1 + temp2 - sc_th0);
  sc_s3 = invK * mod2pi(temp1 + temp2 - sc_thf);
  ok = true;
end

% RSL
function [ok, sc_s1, sc_s2, sc_s3] = RSL(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_th0) + cos(sc_thf);
  S = 2 * sc_Kmax - sin(sc_th0) - sin(sc_thf);
  temp1 = atan2(C, S);
  temp3 = 4 * sc_Kmax^2 - 2 + 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) + sin(sc_thf));
  if (temp3 < 0)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * sqrt(temp3);
  temp2 = atan2(2, sc_s2 * sc_Kmax);
  sc_s1 = invK * mod2pi(sc_th0 - temp1 + temp2);
  sc_s3 = invK * mod2pi(sc_thf - temp1 + temp2);
  ok = true;
end

% RLR
function [ok, sc_s1, sc_s2, sc_s3] = RLR(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_th0) - cos(sc_thf);
  S = 2 * sc_Kmax - sin(sc_th0) + sin(sc_thf);
  temp1 = atan2(C, S);
  temp2 = 0.125 * (6 - 4 * sc_Kmax^2 + 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf)));
  if (abs(temp2) > 1)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * mod2pi(2 * pi - acos(temp2));
  sc_s1 = invK * mod2pi(sc_th0 - temp1 + 0.5 * sc_s2 * sc_Kmax);
  sc_s3 = invK * mod2pi(sc_th0 - sc_thf + sc_Kmax * (sc_s2 - sc_s1));
  ok = true;
end

% LRL
function [ok, sc_s1, sc_s2, sc_s3] = LRL(sc_th0, sc_thf, sc_Kmax);
  invK = 1 / sc_Kmax;
  C = cos(sc_thf) - cos(sc_th0);
  S = 2 * sc_Kmax + sin(sc_th0) - sin(sc_thf);
  temp1 = atan2(C, S);
  temp2 = 0.125 * (6 - 4 * sc_Kmax^2 + 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf)));
  if (abs(temp2) > 1)
    ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
    return;
  end
  sc_s2 = invK * mod2pi(2 * pi - acos(temp2));
  sc_s1 = invK * mod2pi(temp1 - sc_th0 + 0.5 * sc_s2 * sc_Kmax);
  sc_s3 = invK * mod2pi(sc_thf - sc_th0 + sc_Kmax * (sc_s2 - sc_s1));
  ok = true;
end
