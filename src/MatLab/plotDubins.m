%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Demo usage of Dubins shortest path function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reset environment
close all; clear all; clc;


x0=0.0; y0=0.0; th0=1.0;
c=dubinscurve(x0, y0, th0);
figure; axis equal;
plotdubins(c, true, [1 0 0], [0 0 0], [1 0 0]);
c.a1
c.a2
c.a3
c.L

% Implementation of function sinc(t), returning 1 for t==0, and sin(t)/t
% otherwise
function s = sinc(t)
  if (abs(t) < 0.002)
    % For small values of t use Taylor series approximation
    s = 1 - t^2/6 * (1 - t^2/20);
  else
    s = sin(t)/t;
  end
end

% Normalize an angle (in range [0,2*pi))
function out = mod2pi(ang)
  out = ang;
  while (out < 0)
    out = out + 2 * pi;
  end
  while (out >= 2 * pi)
    out = out - 2 * pi;
  end
end


function c = dubinsarc(x0, y0, th0, k, L, i)
  c.x0 = x0;
  c.y0 = y0;
  c.th0 = th0;
  c.k = k;
  c.L = L;
  [c.xf, c.yf, c.thf] = circline(L, x0, y0, th0, k);
  if i==3
    c.thf=pi/2;
  end
end

% Create a structure representing a Dubins curve (composed by three arcs)
function d = dubinscurve(x0, y0, th0)
  d = struct();
  d.a1 = dubinsarc(x0, y0, th0, -1.0, 0.2126801, 1);
  d.a2 = dubinsarc(d.a1.xf, d.a1.yf, d.a1.thf, 0.000000, 140.501550, 2);
  d.a3 = dubinsarc(d.a2.xf, d.a2.yf, d.a2.thf, -1.000000, 0.787320, 3);
  d.L = d.a1.L + d.a2.L + d.a3.L;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Functions to plot Dubins curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluate an arc (circular or straight) composing a Dubins curve, at a 
% given arc-length s
function [x, y, th] = circline(s, x0, y0, th0, k)
  x = x0 + s * sinc(k * s / 2.0) * cos(th0 + k * s / 2);
  y = y0 + s * sinc(k * s / 2.0) * sin(th0 + k * s / 2);
  th = mod2pi(th0 + k * s);
end

% Plot an arc (circular or straight) composing a Dubins curve
function plotarc(arc, color)
  npts = 100;
  pts = zeros(npts+1, 2);
  for j = 0:npts
    s = arc.L/npts * j;
    [x, y] = circline(s, arc.x0, arc.y0, arc.th0, arc.k);
    pts(j+1, 1:2) = [x, y];
  end
  plot(pts(:, 1), pts(:, 2), 'Color', color, 'LineWidth', 4);
end

% Plot a Dubins curve
function plotdubins(curve, decorations, c1, c2, c3)
  currhold = ishold;
  hold on;
  
  % Plot arcs
  plotarc(curve.a1, c1);
  plotarc(curve.a2, c2);
  plotarc(curve.a3, c3);
  
  % Plot initial and final position and orientation
  if decorations
    plot(curve.a1.x0, curve.a1.y0, 'ob');
    plot(curve.a3.xf, curve.a3.yf, 'ob');
    quiver(curve.a1.x0, curve.a1.y0, 0.1*curve.L*cos(curve.a1.th0), 0.1*curve.L*sin(curve.a1.th0), 'b', 'LineWidth', 2);
    quiver(curve.a3.xf, curve.a3.yf, 0.1*curve.L*cos(curve.a3.thf), 0.1*curve.L*sin(curve.a3.thf), 'b', 'LineWidth', 2);
  end
  
  if ~currhold
    hold off;
  end
end
