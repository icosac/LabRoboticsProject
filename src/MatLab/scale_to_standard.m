close all; clear all; clc;

fl= fopen("/Users/enrico/GoogleDrive/Magistrale/1sem/Laboratory of Applied Robotics/LabRoboticsProject/data/test/ML_scale.test", "w");

value=0;
Kmax = 1.0;
x=0; y=0; X=0; Y=0; 
fprintf(fl, "x0, y0, th0, x1, y1, th1, ok, sc_s1, sc_s2, sc_s3\n");
for x0=0.0 : 20 : 150.0
  for y0=0.0 : 20 : 100.0
    for th0=0.0 : 0.25 : 2*pi
      for x1=0.0 : 20 : 150.0
        for y1=0.0 : 20 : 100.0
          for th1=0.0 : 0.25 : 2*pi
            % if (x~=x0 || y~=y0 || X~=x1 || Y~=y1) 
            %   x=x0; y=y0; X=x1; Y=y1; 
            %   fprintf("x0: %f, y0: %f, x1: %f, y1: %f\n", x, y, X, Y);
            % end
            [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, x1, y1, th1, Kmax);
            fprintf(fl, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x0, y0, th0, x1, y1, th1, sc_th0, sc_thf, sc_Kmax, lambda);
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

