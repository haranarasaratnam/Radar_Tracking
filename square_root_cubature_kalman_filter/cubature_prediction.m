% Cubature Prediction
function [x_pred, p_sqrt_pred] = cubature_prediction(x_upd, p_sqrt_upd, Q_sqrt, f)
    [XPts, W] = moments2points(x_upd, p_sqrt_upd);
    n = length(x_upd);
    XpropPts = zeros(n, 2 * n);
    for i = 1:2 * n
        XpropPts(:, i) = f(XPts(:, i));
    end
    x_pred = sum(XpropPts, 2) / (2 * n);
    augmented_mat = [(XpropPts - repmat(x_pred,1,2 * n)) / sqrt(2 * n), Q_sqrt];
    [~, R_qr] = qr(augmented_mat', 0);
    p_sqrt_pred = R_qr';
end