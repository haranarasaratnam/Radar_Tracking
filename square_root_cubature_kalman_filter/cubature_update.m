% Cubature Update
function [x_upd, p_sqrt_upd] = cubature_update(x_pred, p_sqrt_pred, z, R_sqrt, h)
    [XPts, W] = moments2points(x_pred, p_sqrt_pred);
    n = length(x_pred);
    m = length(z);
    ZPts = zeros(m, 2 * n);
    for i = 1:2 * n
        ZPts(:, i) = h(XPts(:, i));
    end
    z_pred = sum(ZPts, 2) / (2 * n);
    augmented_mat = [(ZPts - repmat(z_pred,1,2 * n)) / sqrt(2 * n), R_sqrt; (XPts - repmat(x_pred,1,2 * n)) / sqrt(2 * n), zeros(n, m)];
    [~, R_qr] = qr(augmented_mat', 0);
    R_qr = R_qr';
    T_11 = R_qr(1:m, 1:m);
    T_21 = R_qr(m+1:end, 1:m);
    T_22 = R_qr(m+1:end, m+1:end);
    Gain = T_21/T_11;
    x_upd = x_pred + Gain * (z - z_pred);
    p_sqrt_upd = T_22;
end