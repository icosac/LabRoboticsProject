close all; clear all; clc;

fl= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/LSL_matlab.test", "w");

value=0;
Kmax = 1.0;
x=0; y=0; X=0; Y=0; 
fprintf(fl, "x0, y0, th0, x1, y1, th1, ok, sc_s1, sc_s2, sc_s3\n");
for x0=0.0 : 1 : 150.0
  for y0=0.0 : 1 : 100.0
    for th0=0.0 : 0.1 : 2*pi
      for x1=0.0 : 1 : 150.0
        for y1=0.0 : 1 : 100.0
          for th1=0.0 : 0.1 : 2*pi
            % if (x~=x0 || y~=y0 || X~=x1 || Y~=y1) 
            %   x=x0; y=y0; X=x1; Y=y1; 
            %   fprintf("x0: %f, y0: %f, x1: %f, y1: %f\n", x, y, X, Y);
            % end
            [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, x1, y1, th1, Kmax);
            [ok, sc_s1, sc_s2, sc_s3] = LSL(sc_th0, sc_thf, sc_Kmax);
            fprintf(fl, "%f, %f, %f, %f, %f, %f, %d, ", x0, y0, th0, x1, y1, th1, ok);
            if (ok==true) 
              fprintf(fl, "%f, %f, %f\n", sc_s1, sc_s2, sc_s3);
            else
              fprintf(fl, "0, 0, 0\n");
            end 
          end
        end
      end
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

% RSR
function [ok, sc_s1, sc_s2, sc_s3] = RSR(sc_th0, sc_thf, sc_Kmax)
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