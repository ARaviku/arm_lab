"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from statistics import mean
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rxarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """


    links = np.arange(0, 6, 1)

    if link not in links:
        raise NotImplementedError
    
    if len(joint_angles) != 5:
        raise Exception("Incorrect joint angle length")

    # Add Joint Angles to the dh_params table
    for idx in range(len(dh_params)):
        dh_params[idx][0] += joint_angles[idx]

    # Slice the dh_params based on the link number
    dh_params_link = dh_params[:link]
    T = np.eye(4)

    # Iterate through dh_params to get the final end-effector position
    for i in range(len(dh_params_link)):
        # print(get_transform_from_dh(dh_params_link[i][0], dh_params_link[i][1], dh_params_link[i][2], dh_params_link[i][3]))
        # print(T)
        T = T @ get_transform_from_dh(dh_params_link[i][0], dh_params_link[i][1], dh_params_link[i][2], dh_params_link[i][3])
        # print(f'Transform{T}')

    phi, theta, psi = get_euler_angles_from_T(T)
    x, y, z = T[0][3], T[1][3], T[2][3]

    out = [x, y, z, phi, theta, psi]

    return out


def get_transform_from_dh(theta, d, a, alpha):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    rotation_matrix = T[:3, :3]
    # print(f'rotation_matrix {rotation_matrix}')
    # Convert the rotation matrix to Euler angles.
    r = Rotation.from_matrix(rotation_matrix)
    # print(f'r{r}')
    euler_angles = r.as_euler('zyx', degrees=False)
    # print(f"euler angles {euler_angles}")
    return euler_angles[0], euler_angles[1], euler_angles[2]

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.

    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass