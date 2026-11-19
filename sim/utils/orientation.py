import numpy as np


def compute_alpha(epsilon):
    """
    Compute orientation alignment with gravity.

    epsilon: quaternion (x, y, z, w)
    returns: alpha in [0, 1]
    """

    qx, qy, qz, qw = epsilon

    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    g_world = np.array([0.0, 0.0, -1.0])
    g_body = R.T @ g_world

    alpha = -g_body[2]
    return np.clip(alpha, -1.0, 1.0)


def gravity_body(epsilon):
    """
    Gravity direction expressed in the body frame (R^-1 . g).

    epsilon: quaternion (x, y, z, w)
    returns: unit vector (3,) -> [0, 0, -1] when the robot is upright
    """

    qx, qy, qz, qw = epsilon

    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    return R.T @ np.array([0.0, 0.0, -1.0])
