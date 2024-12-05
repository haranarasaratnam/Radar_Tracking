% Cubature Kalman Filter
function [x_upd, p_sqrt_upd] = cubature_kalman_filter(x_est, p_sqrt_est, z, Q_sqrt, R_sqrt, f, h)
    [x_pred, p_sqrt_pred] = cubature_prediction(x_est, p_sqrt_est, Q_sqrt, f);
    [x_upd, p_sqrt_upd] = cubature_update(x_pred, p_sqrt_pred, z, R_sqrt, h);
end