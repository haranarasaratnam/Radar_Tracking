% Spherical-Radial Transform
function [CPts, Weights] = moments2points(mu, p_sqrt)
    nx = length(mu);
    Weights = ones(1, 2 *nx) / (2 * nx);
    CPts = repmat(mu, 1, 2 * nx) + sqrt(nx) * p_sqrt * [eye(nx) -eye(nx)];
end
