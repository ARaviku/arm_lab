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
    euler_angles = r.as_euler('ZYZ', degrees=False)
    # print(f"euler angles {euler_angles}")
    return euler_angles[0], euler_angles[1], euler_angles[2]

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.

    """
    Pose = np.zeros((6,1))
    Pose[:3] = T[:3, 3]
    Pose[3:] = Rotation.from_matrix("ZYZ", T[:3, :3])
    return Pose

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


def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      pose       The desired pose vector as np.array  |  pose: [x, y, z , alpha, beta, gamma] (ZYZ Euler) shape (6x1)

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    joint_config = []
    l1 = 103.91
    l2 = 200
    l3 = 50
    l4 = 200
    l5 = 174.15
    end_position = pose[:3]
    end_euler = pose[3:]
    end_matrix = Rotation.from_euler('ZYZ',end_euler.reshape(3)).as_matrix()
    
    # q1,q2,q3,q4,q5 = (0,0,0,0,0)
    o4_origin = end_position - l5 * end_matrix @ np.array([0,0,1]).reshape(3,1)
    q1 = -np.arctan2(o4_origin[0][0], o4_origin[1][0])
    x4p = np.linalg.norm(o4_origin[0:2])
    z4p = o4_origin[2][0] - l1
    theta0 = np.arctan(l3/l2)
    theta1 = np.arctan(z4p/x4p)
    # First Solution
    q3 = np.pi / 2 + theta0 - np.arccos((l2*l2 + l3*l3 + l4*l4 - x4p*x4p - z4p*z4p)/(2*l4*np.sqrt(l2*l2+l3*l3))) # Maybe multiple solution
    theta2 = np.arctan((l3 + l4*np.cos(q3))/(l2 - l4*np.sin(q3)))
    q2 = np.pi / 2 - theta1 - theta2
    R4w = Rotation.from_euler('ZYZ', [q1, - np.pi / 2, np.pi/2 + q2 + q3]).as_matrix()
    Re4 = np.linalg.inv(R4w) @ end_matrix
    Re4a = Rotation.from_matrix(Re4).as_euler('ZYZ')
    q4 = Re4a[0]
    q5 = Re4a[2] - np.pi / 2
    joint_config.append(np.array([q1, q2, q3, q4, q5]))
    # Second Solution
    q3_ = np.pi / 2 + theta0 + np.arccos((l2*l2 + l3*l3 + l4*l4 - x4p*x4p - z4p*z4p)/(2*l4*np.sqrt(l2*l2+l3*l3))) # Maybe multiple solution
    theta2 = np.arctan((l3 + l4*np.cos(q3_))/(l2 - l4*np.sin(q3_)))
    q2_ = np.pi / 2 - theta1 - theta2
    R4w_ = Rotation.from_euler('ZYZ', [q1, - np.pi / 2, np.pi/2 + q2_ + q3_]).as_matrix()
    Re4_ = np.linalg.inv(R4w_) @ end_matrix
    Re4a_ = Rotation.from_matrix(Re4_).as_euler('ZYZ')
    q4_ = Re4a_[0]
    q5_ = Re4a_[2] - np.pi / 2
    joint_config.append(np.array([q1, q2_, q3_, q4_, q5_]))

    return joint_config
