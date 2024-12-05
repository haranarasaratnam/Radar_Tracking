% Cubature Kalman Filter with CTRV Model
% Reference: https://ieeexplore.ieee.org/document/4982682

clear; clc;

% Parameters
dt = 0.1;   % Time step [s]
N = 300;    % Number of steps

show_final = 1;
show_animation = 0;
show_ellipse = 0;

loc_radar = [0; 0];

z_noise = [0.1, 0.0, 0.0; ...              % Range noise [m]
           0.0, deg2rad(1.2), 0.0; ...     % Azimuth noise [rad]
           0.0, 0.0, 0.1];                 % Range rate noise [m/s]

x_0 = [100.0; ...                          % X position [m]
       -30.0; ...                          % X velocity [m/s]
       100.0; ...                          % Y position [m]
       30.0; ...                           % Y velocity [m/s]
       deg2rad(-3)];                       % Turn rate [rad/s]

P_0 = diag([1e1, 1e1, 1e1, 1e1, deg2rad(1)]);

sigma_v1 = 1e-3;
sigma_v2 = deg2rad(1e-2);

G = [dt^2/2, 0, 0; ...
     dt, 0, 0; ...
     0, dt^2/2, 0; ...
     0, dt, 0; ...
     0, 0, dt];

sigma_v = diag([sigma_v1, sigma_v1, sigma_v2]);

Q = G * sigma_v^2 * G';
Q_sqrt = chol(Q, 'lower');

R_sqrt = [0.15, 0.0, 0.0; ...
          0.0, deg2rad(2.5), 0.0; ...
          0.0, 0.0, 0.15];

R = R_sqrt^2;

% Motion Model
f = @(x) [x(1) + x(2) * sin(x(5) * dt) / x(5) - x(4) * (1 - cos(x(5) * dt)) / x(5); ...
          x(2) * cos(x(5) * dt) - x(4) * sin(x(5) * dt); ...
          x(3) + x(4) * sin(x(5) * dt) / x(5) + x(2) * (1 - cos(x(5) * dt)) / x(5); ...
          x(4) * cos(x(5) * dt) + x(2) * sin(x(5) * dt); ...
          x(5)];

% Measurement Model
h = @(x) [sqrt((x(1) - loc_radar(1))^2 + (x(3) - loc_radar(2))^2); ...
          atan2(x(3) - loc_radar(2), x(1) - loc_radar(1)); ...
          ((x(1) - loc_radar(1)) * x(2) + (x(3) - loc_radar(2)) * x(4)) / sqrt((x(1) - loc_radar(1))^2 + (x(3) - loc_radar(2))^2)];

% Generate Measurement
generate_measurement = @(x_true) h(x_true) + z_noise * randn(3, 1);

% Main Simulation
x_est = x_0;
p_sqrt_est = chol(P_0, 'lower');
x_true = x_0;
x_true_cat = [];
x_est_cat = [];
conf_est_cat = [];

for i = 1:N
    x_true = f(x_true);
    z = generate_measurement(x_true);
    [x_est, p_sqrt_est] = cubature_kalman_filter(x_est, p_sqrt_est, z, Q_sqrt, R_sqrt, f, h);

    x_true_cat = [x_true_cat; x_true([1, 3])'];
    x_est_cat = [x_est_cat; x_est([1, 3])'];
    p_est = p_sqrt_est * p_sqrt_est';
    sigma_err_bound = 2 * [sqrt(p_est(1,1)), sqrt(p_est(3,3))];
    conf_est_cat = [conf_est_cat; sigma_err_bound];
end

% Plot Results
if show_final
    figure;
    plot(x_true_cat(:, 1), x_true_cat(:, 2), 'r--'); hold on;
    plot(x_est_cat(:, 1), x_est_cat(:, 2), 'b');
    legend('True Position', 'Estimated Position');
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    title('Cubature Kalman Filter - CTRV Model');
    grid on;

    % Data Preparation
    t_arr = dt * (0:N-1);  % Time array
    x_err = x_true_cat(:, 1) - x_est_cat(:, 1);  % x-position error
    y_err = x_true_cat(:, 2) - x_est_cat(:, 2);  % y-position error
    
    % Create figure and axes for subplots
    fig2 = figure;
    ax(1) = subplot(2, 1, 1);  % First subplot
    ax(2) = subplot(2, 1, 2);  % Second subplot
    
    % Plotting in ax(1)
    plot(ax(1), t_arr, x_err, 'b-', 'LineWidth', 1);
    hold(ax(1), 'on');
    plot(ax(1), t_arr, conf_est_cat(:, 1), 'r--', 'LineWidth', 1);
    plot(ax(1), t_arr, -conf_est_cat(:, 1), 'r--', 'LineWidth', 1);
    grid(ax(1), 'on');
    %ax(1).YLim = [-1, 1];  % Uncomment to set y-axis limits
    ax(1).XLim = [0, N*dt];  % Set x-axis limits
    ylabel(ax(1), 'Error in x pos [m]');
    title(ax(1), 'Error with 2$\sigma$ bound', 'Interpreter', 'latex');
    hold(ax(1), 'off');
    
    % Plotting in ax(2)
    plot(ax(2), t_arr, y_err, 'b-', 'LineWidth', 1);
    hold(ax(2), 'on');
    plot(ax(2), t_arr, conf_est_cat(:, 2), 'r--', 'LineWidth', 1);
    plot(ax(2), t_arr, -conf_est_cat(:, 2), 'r--', 'LineWidth', 1);
    grid(ax(2), 'on');
    %ax(2).YLim = [-1, 1];  % Uncomment to set y-axis limits
    ax(2).XLim = [0, N*dt];  % Set x-axis limits
    ylabel(ax(2), 'Error in y pos [m]');
    xlabel(ax(2), 'Time [s]');
    hold(ax(2), 'off');
    
    % Adjust figure properties
    set(fig2, 'Position', [100, 100, 600, 400]);  % Adjust figure size (width, height)
    
    % Display the figure
    drawnow;



end
