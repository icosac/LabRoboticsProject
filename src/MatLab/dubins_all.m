close all; clear all; clc;

LSLf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/LSL_matlab.test", "w");
RSRf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/RSR_matlab.test", "w");
LSRf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/LSR_matlab.test", "w");
RSLf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/RSL_matlab.test", "w");
RLRf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/RLR_matlab.test", "w");
LRLf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/LRL_matlab.test", "w");
Coorf= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/coordinates_matlab.test", "w");

value=0;
Kmax = 1.0;
x=0; y=0; X=0; Y=0; 
for x0=0.0 : 4 : 150.0
  for y0=0.0 : 4 : 100.0
    for th0=0.0 : 0.4 : 2*pi
      for x1=0.0 : 4 : 150.0
        for y1=0.0 : 4 : 100.0
          for th1=0.0 : 0.4 : 2*pi
            if (x~=x0 || y~=y0 || X~=x1 || Y~=y1) 
              x=x0; y=y0; X=x1; Y=y1; 
              % fprintf("x0: %f, y0: %f, x1: %f, y1: %f\n", x, y, X, Y);
            end
            fprintf(Coorf, "%f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1);
            [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, x1, y1, th1, Kmax);
           
            primitives = {@LSL_, @RSR_, @LSR_, @RSL_, @RLR_, @LRL_};
            names = {LSLf, RSRf, LSRf, RSLf, RLRf, LRLf};

            func_(primitives, names, sc_th0, sc_thf, sc_Kmax, lambda) 
         
          end
        end
      end
    end
  end
end

function func_ (primitives, names, sc_th0, sc_thf, sc_Kmax, lambda)
  for i = 1:numel(primitives)
    [ok, sc_s1, sc_s2, sc_s3] = primitives{i}(sc_th0, sc_thf, sc_Kmax);
    if (ok==true) 
      fprintf(names{i}, "%f, %f, %f\n", sc_s1, sc_s2, sc_s3);
    else
      fprintf(names{i}, "0\n");
    end
  end
end


function out = mod2pi(ang)
  out = ang;
  while (out < 0)
    out = out + 2 * pi;
  end
  while (out >= 2 * pi)
    out = out - 2 * pi;
  end
end

function [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, xf, yf, thf, Kmax)
  % find transform parameters
  dx = xf - x0;
  dy = yf - y0;
  phi = atan2(dy, dx);
  lambda = hypot(dx, dy);

  C = dx / lambda;
  S = dy / lambda;
  lambda = lambda / 2;

  % scale and normalize angles and curvature
  sc_th0 = mod2pi(th0 - phi);
  sc_thf = mod2pi(thf - phi);
  sc_Kmax = Kmax * lambda;
end

function [ok, sc_s1, sc_s2, sc_s3] = LSL_(sc_th0, sc_thf, sc_Kmax)
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
function [ok, sc_s1, sc_s2, sc_s3] = RSR_(sc_th0, sc_thf, sc_Kmax)
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
function [ok, sc_s1, sc_s2, sc_s3] = LSR_(sc_th0, sc_thf, sc_Kmax)
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
function [ok, sc_s1, sc_s2, sc_s3] = RSL_(sc_th0, sc_thf, sc_Kmax)
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
function [ok, sc_s1, sc_s2, sc_s3] = RLR_(sc_th0, sc_thf, sc_Kmax)
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
function [ok, sc_s1, sc_s2, sc_s3] = LRL_(sc_th0, sc_thf, sc_Kmax)
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
