

"""
Cubature Kalman filter using Constant Turn Rate and Velocity (CTRV) model

https://ieeexplore.ieee.org/document/4982682

Author: Haran Arasaratnam

state matrix:                       2D x-y position, yaw, velocity and yaw rate
measurement matrix:                 2D range and azimuth angle

dt:                                 Duration of time step
N:                                  Number of time steps
show_final:                         Flag for showing final result
show_animation:                     Flag for showing each animation frame
show_ellipse:                       Flag for showing covariance ellipse
z_noise:                            Measurement noise
x_0:                                Prior state estimate matrix
P_0:                                Prior state estimate covariance matrix
Q:                                  Process noise covariance
hx:                                 Measurement model matrix
R:                                  Sensor noise covariance
CP:                                 Cubature Points
W:                                  Weights

x_est:                              State estimate
P_est:                              State estimate covariance
x_true:                             Ground truth value of state
x_true_cat:                         Concatenate all ground truth states
x_est_cat:                          Concatenate all state estimates
z_cat:                              Concatenate all measurements

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

dt = 0.1
N = 300

show_final = 1
show_animation = 0
show_ellipse = 0

loc_radar = np.array([0, 0])


z_noise = np.array([[0.1, 0.0, 0.0],             # range   [m]
                    [0.0, np.deg2rad(1.2), 0.0], # azimuth [rad]
                    [0.0, 0.0, 0.1]])           # range rate [m/s]




x_0 = np.array([[100.0],                                  # x position    [m]
                [-30.0],                                  # x velocity     [m/s]
                [100.0],                                  # x position    [m]
                [30.0],                                   # y velocity      [m/s]
                [np.deg2rad(-3)]])                        # turn rate     [rad/s]


p_0 = np.array([[1e1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1e1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1e1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1e1, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.deg2rad(1)]])


sigma_v1 = 1e-3
sigma_v2 = np.deg2rad(1e-2)

G = np.array([[dt**2/2, 0, 0],
              [dt, 0, 0],
              [0, dt**2/2, 0],
              [0, dt, 0],
              [0, 0, dt]])

sigma_v = np.diag([sigma_v1, sigma_v1, sigma_v2])

Q = G @ sigma_v ** 2 @ G.T


R = np.array([[0.15, 0.0, 0.0],
              [0.0, np.deg2rad(2.5), 0.0],
              [0.0, 0.0, 0.15]])**2



def f(x):
    """
    Motion Model
    References:
    https://www.mathworks.com/help/fusion/ug/motion-model-state-and-process-noise.html
    """

    x[0] = x[0] + x[1] * np.sin(x[4]*dt)/x[4] - x[3] * (1- (np.cos(x[4]*dt)))/x[4]
    x[1] = x[1] * np.cos(x[4]*dt) - x[3]* np.sin(x[4]*dt)
    x[2] = x[2] + x[3] * np.sin(x[4]*dt)/x[4] + x[1] * (1- (np.cos(x[4]* dt)))/x[4]
    x[3] = x[3] * np.cos(x[4]*dt) + x[1] * np.sin(x[4]*dt)
    x[4] = x[4]

    return x


def h(x):
    """Measurement Model
    Range, azimuth angle and turn rate
    """
    z = np.empty((3,1))
    dx = x[0] - loc_radar[0]
    dy = x[2] - loc_radar[1]
    z[0] = math.sqrt(dx ** 2 + dy ** 2)
    z[1] = np.arctan2(dy, dx)
    z[2] = (dx * x[1] + dy * x[3]) / z[0]
    return z

def generate_measurement(x_true):
    gz = h(x_true)
    z = gz + z_noise @ np.random.randn(3, 1)
    return z


def moments2points(mu, P):
    """
    Spherical-Radial Transform using Cubature Rule
    Generate 2n Cubature Points to represent the nonlinear model
    Assign Weights to each Cubature Point, Wi = 1/2n
    """

    n_dim = len(mu)
    weights = np.ones(2 * n_dim) / (2*n_dim)
    sigma = linalg.cholesky(P,lower=True)
    points = np.tile(mu, (1, 2 * n_dim))
    points[:, 0:n_dim] += sigma * np.sqrt(n_dim)
    points[:, n_dim:] -= sigma * np.sqrt(n_dim)
    return points, weights


def cubature_prediction(x_upd, p_upd):
    n = len(x_upd)
    [CP, W] = moments2points(x_upd, p_upd)
    x_pred = np.zeros((n, 1))
    p_pred = Q
    for i in range(2*n):
        x_pred = x_pred + (f(CP[:, i]).reshape((n, 1)) * W[i])
    for i in range(2*n):
        p_step = (f(CP[:, i]).reshape((n, 1)) - x_pred)
        p_pred = p_pred + (p_step @ p_step.T * W[i])
    return x_pred, p_pred


def cubature_update(x_pred, p_pred, z):
    n, m = len(x_pred), len(z)
    [CP, W] = moments2points(x_pred, p_pred)
    z_pred = np.zeros((m, 1))
    P_xy = np.zeros((n, m))
    P_zz = R
    for i in range(2*n):
        z_pred = z_pred + (h(CP[:, i]).reshape((m, 1)) * W[i])
    for i in range(2*n):
        p_step = (h(CP[:, i]).reshape((m, 1)) - z_pred)
        P_xy = P_xy + ((CP[:, i]).reshape((n, 1)) -
                       x_pred) @ p_step.T * W[i]
        P_zz = P_zz + p_step @ p_step.T * W[i]
    x_upd = x_pred + P_xy @ np.linalg.pinv(P_zz) @ (z - z_pred)
    p_upd = p_pred - P_xy @ np.linalg.pinv(P_zz) @ P_xy.T
    return x_upd, p_upd


def cubature_kalman_filter(x_est, p_est, z):
    x_pred, p_pred = cubature_prediction(x_est, p_est)
    x_upd, p_upd = cubature_update(x_pred, p_pred, z)
    return x_upd, p_upd




def plot_animation(i, x_true_cat, x_est_cat):
    if i == 0:
        pass
        # plt.plot(x_true_cat[0], x_true_cat[1], '.r')
        # plt.plot(x_est_cat[0], x_est_cat[1], '.b')
    else:
        plt.plot(x_true_cat[1:, 0], x_true_cat[1:, 1], 'r--', label='True')
        plt.plot(x_est_cat[1:, 0], x_est_cat[1:, 1], 'b', label='CKF')
    plt.pause(0.001)


def plot_ellipse(x_est, p_est):
    phi = np.linspace(0, 2 * math.pi, N)
    p_ellipse = np.array(
        [[p_est[0, 0], p_est[0, 1]], [p_est[1, 0], p_est[1, 1]]])
    x0 = 2 * linalg.sqrtm(p_ellipse)
    xy_1 = np.array([])
    xy_2 = np.array([])
    for i in range(N):
        arr = np.array([[math.sin(phi[i])], [math.cos(phi[i])]])
        arr = x0 @ arr
        xy_1 = np.hstack([xy_1, arr[0]])
        xy_2 = np.hstack([xy_2, arr[1]])
    plt.plot(xy_1 + x_est[0], xy_2 + x_est[1], 'r', linewidth=0.5)
    plt.grid(True)
    plt.pause(0.00001)


def plot_final(x_true_cat, x_est_cat, conf_est_cat):
    fig1, ax = plt.subplots()
    ax.plot(x_true_cat[0:, 0], x_true_cat[0:, 1],
                 'r--', label='True Position')
    ax.plot(x_est_cat[0:, 0], x_est_cat[0:, 1],
                 'b', label='Estimated Position')
    ax.plot(loc_radar[0], loc_radar[1], marker='D', color='red', markersize=5, linestyle='None', label='Radar')
    ax.plot(x_true_cat[0,0], x_true_cat[0,1], marker='*', color='red', markersize=5, linestyle='None', label='Start')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Cubature Kalman Filter - CTRV Model')
    ax.legend(loc='best', shadow=True, fontsize='small')
    # ax.set_xlim([0,12])
    # ax.set_ylim([0,25])
    plt.grid(True)


    fig2, ax = plt.subplots(2,1)
    t_arr = dt * np.arange(N)
    x_err = x_true_cat[0:, 0] - x_est_cat[0:, 0]
    y_err = x_true_cat[0:, 1] - x_est_cat[0:, 1]

    ax[0].plot(t_arr, x_err, 'b-', linewidth=1)
    ax[0].plot(t_arr, conf_est_cat[0:,0], 'r--', linewidth=1)
    ax[0].plot(t_arr, -1 * conf_est_cat[0:,0], 'r--', linewidth=1)
    ax[0].grid(True)
    #ax[0].set_ylim([-1,1])
    ax[0].set_xlim([0, int(N*dt)])
    ax[0].set_ylabel('Error in x pos [m]')
    ax[0].set_title(r'Error with 2$\sigma$ bound')

    ax[1].plot(t_arr, y_err, 'b-', linewidth=1)
    ax[1].plot(t_arr, conf_est_cat[0:,1], 'r--', linewidth=1)
    ax[1].plot(t_arr, -1 * conf_est_cat[0:,1], 'r--', linewidth=1)

    ax[1].grid(True)
    #ax[1].set_ylim([-1,1])
    ax[1].set_xlim([0, int(N*dt)])
    ax[1].set_ylabel('Error in y pos [m]')
    ax[1].set_xlabel('Time [s]')

    plt.show()


def main():
    print(__file__ + " start!!")
    x_est = x_0
    p_est = p_0
    x_true = x_0
    x_true_cat = np.empty((2,))
    x_est_cat = np.empty((2,))
    conf_est_cat = np.empty((2,))
    for i in range(N):
        x_true = f(x_true)
        z = generate_measurement(x_true)
        x_est, p_est = cubature_kalman_filter(x_est, p_est, z)

        x_true_cat = np.vstack((x_true_cat, x_true[0:3:2].T))
        x_est_cat = np.vstack((x_est_cat, x_est[0:3:2].T))
        sigma_err_bound = np.array([2 * math.sqrt(p_est[0,0]), 2 * math.sqrt(p_est[2,2])])
        conf_est_cat = np.vstack((conf_est_cat, sigma_err_bound))


        if i == (N - 1) and show_final == 1:
            show_final_flag = 1
        else:
            show_final_flag = 0
        if show_animation == 1:
            plot_animation(i, x_true_cat, x_est_cat)
        if show_ellipse == 1:
            plot_ellipse(x_est[0:2], p_est)


        if show_final_flag == 1:
            x_true_cat = x_true_cat[1:,]
            x_est_cat = x_est_cat[1:,]
            conf_est_cat = conf_est_cat[1:,]
            plot_final(x_true_cat, x_est_cat, conf_est_cat)

    print('CKF Done!!')


if __name__ == '__main__':
    main()
