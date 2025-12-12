
"""
PybulletRobot
~~~~~~~~~~~~~
"""

import sys
import os
import yaml
import pickle
import time
import datetime

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin

from ..utils import *

from scipy.spatial.transform import Rotation as R

JOINT_SAFETY_FACTOR = 0.95




def get_subdirectories(path):
    return [os.path.basename(f.path) for f in os.scandir(path) if f.is_dir()]


class PybulletRobot:
    """
    Pybullet Simulator Robot Class

    :param int ClientId: pybullet GUI client's ID
    :param dict[] robot_info: dictionary of robot's information
    :param float dt: simulation time step
    """
    def __init__(self, ClientId, robot_info, dt):

        # Simulator configuration
        self.__filepath = os.path.dirname(os.path.abspath(__file__))
        self.__urdfpath = self.__filepath + "/../assets/urdf"

        self.ClientId = ClientId
        self._robot_info = robot_info
        self.dt = dt

        self._initialization()

    def robot_update(self):
        """
        Update the state of the robot by implementing _pre_robot_update, _get_robot_states, _compute_torque_input,
        _control_robot, and _post_robot_update methods.
        """

        ##custom
        if hasattr(self, "time"):
            self.time += self.dt



        self._pre_robot_update()      # pre update

        self._get_robot_states()      # update robot's states
        self._compute_torque_input()  # compute applied motor torques
        self._control_robot()         # apply motor torques

        self._post_robot_update()     # post update

    def _pre_robot_update(self):
        """
        protected method used by robot_update.
        """
        pass

    def _post_robot_update(self):
        """
        protected method used by robot_update.
        """

        if self._is_constraint_visualization:
            self._constraint_visualizer()


    def _initialization(self):
        """
        Protected method _initialize sets all data to None, False, or emptiness. Then finally this method loads robot by
        implementing protected method _load_robot().
        """

        # Load robot
        self._import_robot()
        self._init_robot_parameters()

        PRINT_BLUE("******** ROBOT INFO ********")
        PRINT_BLACK("Robot name", self.robot_name)
        PRINT_BLACK("Robot type", self.robot_type)
        PRINT_BLACK("DOF", self.numJoints)
        PRINT_BLACK("Joint limit", self._is_joint_limit)
        PRINT_BLACK("Constraint visualization", self._is_constraint_visualization)
        PRINT_RED("debug", self.RobotEEJointIdx)
        PRINT_RED("debuh2",self._ee_frame_id)

    def _import_robot(self):
        """
        This method is protected method of pybulletRobot class which load robot's information from yaml file
        """

        # Get robot configuration
        self._robot_name = self._robot_info["robot_name"]              # robot name
        self._base_pos = self._robot_info["robot_position"]            # base position
        base_eul = self._robot_info["robot_orientation"]               # base orientation (euler XYZ, degree)
        self._base_quat = eul2quat(base_eul, 'XYZ', degree=True)       # base orientation (quaternion)
        self._base_SE3 = xyzquat2SE3(self._base_pos, self._base_quat)  # base SE3

        self._is_joint_limit = self._robot_info["robot_properties"]["joint_limit"]
        self._is_constraint_visualization = self._robot_info["robot_properties"]["constraint_visualization"]

        # Search urdf file
        available_robot_types = get_subdirectories(self.__urdfpath)
        load_success = False
        for robot_type in available_robot_types:
            available_robot_names = get_subdirectories(self.__urdfpath + "/{}".format(robot_type))
            if self.robot_name in available_robot_names:
                self._robot_type = robot_type  # robot type
                load_success = True
                break

        if not load_success:
            PRINT_RED("*** NO AVAILABLE ROBOT ***")
            PRINT_BLACK("Robot name", self.robot_name)
            return

        try:
            # Open YAML file
            with open(self.__urdfpath + "/{}/robot_configs.yaml".format(self._robot_type)) as yaml_file:
                self._robot_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)
        except:
            PRINT_RED("*** FAILED TO LOAD ROBOT CONFIGS ***")
            PRINT_BLACK("Robot name", self.robot_name)
            PRINT_BLACK("Robot type", self.robot_type)
            return

        # Import robot in PyBullet
        flags = p.URDF_USE_INERTIA_FROM_FILE + p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        urdf_dir = self.__urdfpath + "/{0}/{1}".format(self.robot_type, self.robot_name)
        urdf_path = urdf_dir + "/model.urdf"

        self.robotId = p.loadURDF(urdf_path, basePosition=self._base_pos, baseOrientation=self._base_quat,
                                  flags=flags, physicsClientId=self.ClientId, useFixedBase=1)


        # Get robot's properties from the robot_info.yaml file
        self.RobotBaseJointIdx = self._robot_configs[self.robot_name]["JointInfo"]["RobotBaseJoint"]
        self.RobotMovableJointIdx = self._robot_configs[self.robot_name]["JointInfo"]["RobotMovableJoint"]
        self.RobotEEJointIdx = self._robot_configs[self.robot_name]["JointInfo"]["RobotEEJoint"]
        self.RobotFTJointIdx = self._robot_configs[self.robot_name]["JointInfo"]["RobotFTJoint"]

        if len(self.RobotBaseJointIdx) == 0:
            self.RobotBaseJointIdx = [-1]
        if len(self.RobotEEJointIdx) == 0:
            self.RobotEEJointIdx = [self.RobotMovableJointIdx[-1]]

        # Get robot base's pose
        state = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.ClientId)
        T_wg = xyzquat2SE3(state[0], state[1]) # world to ground
        if self.RobotBaseJointIdx[0] == -1:
            state = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.ClientId)
        else:
            state = p.getLinkState(self.robotId, self.RobotBaseJointIdx[0], physicsClientId=self.ClientId)
        T_wb = xyzquat2SE3(state[0], state[1]) # world to base

        self._T_gb = TransInv(T_wg) @ T_wb  # pose of robot's base frame in robot's ground frame

        # Get pinocchio model to compute robot's dynamics & kinematics
        self.pinModel = PinocchioModel(urdf_dir, self._base_SE3 @ self._T_gb)


        ### custom
        #self._model = self.pinModel.pinModel      # pinocchio.Model
        #self._data  = self.pinModel.pinData       # pinocchio.Data

        self._ee_frame_id = self.pinModel.pinModel.getFrameId("panda_ee")
        #self._ee_frame_id = self.pinModel.pinModel.nframes - 1
    



        # set robot's number of bodies and number of joints
        self._numBodies = 1 + p.getNumJoints(self.robotId, self.ClientId)
        self._numJoints = len(self.RobotMovableJointIdx)

        for idx in self.RobotFTJointIdx:
            p.enableJointForceTorqueSensor(self.robotId, jointIndex=idx, physicsClientId=self.ClientId)

        # Unlock joint limit
        if self._is_joint_limit is False:
            for idx in self.RobotMovableJointIdx:
                p.changeDynamics(self.robotId, idx, jointLowerLimit=-314, jointUpperLimit=314,
                                 physicsClientId=self.ClientId)

        # Get robot color
        self._robot_color = [None] * self.numBodies
        visual_data = p.getVisualShapeData(self.robotId)
        for data in visual_data:
            self._robot_color[data[1]+1] = data[7]

        # Add end-effector coordinate frame visualizer
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 0.7],
                                            physicsClientId=self.ClientId)

        self._endID = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, 0],
                                        baseOrientation=[0, 0, 0, 1], physicsClientId=self.ClientId)

        self._endID_x = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.05, 0, 0], lineColorRGB=[1, 0, 0],
                                           lineWidth=2, parentObjectUniqueId=self._endID,
                                           physicsClientId=self.ClientId)
        self._endID_y = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.05, 0], lineColorRGB=[0, 1, 0],
                                           lineWidth=2, parentObjectUniqueId=self._endID,
                                           physicsClientId=self.ClientId)
        self._endID_z = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.05], lineColorRGB=[0, 0, 1],
                                           lineWidth=2, parentObjectUniqueId=self._endID,
                                           physicsClientId=self.ClientId)

        # Remove the PyBullet's built-in position controller's effect
        p.setJointMotorControlArray(bodyUniqueId=self.robotId,
                                    jointIndices=self.RobotMovableJointIdx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * self.numJoints,
                                    forces=[0] * self.numJoints,
                                    physicsClientId=self.ClientId
                                    )

    @property
    def robot_name(self):
        return self._robot_name

    @property
    def robot_type(self):
        return self._robot_type

    @property
    def base_pos(self):
        return self._base_pos[:]

    @property
    def base_quat(self):
        return self._base_quat[:]

    @property
    def base_SE3(self):
        return self._base_SE3.copy()

    @property
    def numJoints(self):
        return self._numJoints

    @property
    def numBodies(self):
        return self._numBodies

    # Get robot's information
    def _init_robot_parameters(self):
        """
        Initialize robot state variables and dynamic parameters
        """
        # Robot's states
        self._q = np.zeros([self.numJoints, 1])        # position of joints (rad)
        self._qdot = np.zeros([self.numJoints, 1])     # velocity of joints (rad/s)
        self._qddot = np.zeros([self.numJoints, 1])    # acceleration of joints (rad/s^2)

        self._q_des = np.zeros([self.numJoints, 1])      # desired position of joints
        self._qdot_des = np.zeros([self.numJoints, 1])   # desired velocity of joints
        self._qddot_des = np.zeros([self.numJoints, 1])  # desired acceleration of joints

        self._p_des = np.zeros([3,1])
        self._R_des = np.zeros([3,1])

        self._Js = np.zeros([6, self.numJoints])      # Spatial jacobian matrix.
        self._Jb = np.zeros([6, self.numJoints])      # Body jacobian matrix.
        self._Jr = np.zeros([6, self.numJoints])      # Jacobian matrix.
        self._Jsinv = np.zeros([self.numJoints, 6])   # Inverse of spatial jacobian matrix
        self._Jbinv = np.zeros([self.numJoints, 6])   # Inverse of body jacobian matrix
        self._Jrinv = np.zeros([self.numJoints, 6])   # Inverse of jacobian matrix

        self._M = np.zeros([self.numJoints, self.numJoints])  # Mass matrix of robot
        self._C = np.zeros([self.numJoints, self.numJoints])  # Coriolis matrix of robot
        self._c = np.zeros([self.numJoints, 1])               # Coriolis vector of robot (c = C@qdot)
        self._g = np.zeros([self.numJoints, 1])               # Gravity vector of robot
        self._tau = np.zeros([self.numJoints, 1])             # Input torque (N*m)
        self._iq_err = np.zeros([self.numJoints, 1])

        self._p = np.zeros([6, 1])       # End-effector's pose (xyz, xi)
        self._T_end = np.zeros([4, 4])   # End-effector's pose in SE3

        self._ft = np.zeros([6, 1])  # F/T value
        self._ft_bias = np.zeros([6, 1]) 
        self._is_ft_calibrated = False  
        self._ft_filtered = np.zeros([6, 1])
        self._alpha = 0.05

        # Constraint & flag
        self._jointpos_lower = [0 for _ in range(self.numJoints)]
        self._jointpos_upper = [0 for _ in range(self.numJoints)]
        self._jointvel = [0 for _ in range(self.numJoints)]
        self._jointforce = [0 for _ in range(self.numJoints)]

        self._jointpos_flag = [0 for _ in range(self.numJoints)]
        self._jointvel_flag = [0 for _ in range(self.numJoints)]
        self._jointforce_flag = [0 for _ in range(self.numJoints)]
        self._collision_flag = [0 for _ in range(self.numBodies)]

        # Get joint constraints
        for idx in range(self.numJoints):
            jointInfo = p.getJointInfo(bodyUniqueId=self.robotId, jointIndex=self.RobotMovableJointIdx[idx],
                                       physicsClientId=self.ClientId)
            self._jointpos_lower[idx] = jointInfo[8] * JOINT_SAFETY_FACTOR
            self._jointpos_upper[idx] = jointInfo[9] * JOINT_SAFETY_FACTOR
            self._jointvel[idx] = jointInfo[10] * JOINT_SAFETY_FACTOR
            self._jointforce[idx] = jointInfo[11] * JOINT_SAFETY_FACTOR



        self.control_mode = "idc"



        # custom1 : 안정화
        self._get_robot_states()  # 현재 상태 읽어오기
        self._q_des = self._q.copy()
        self._qdot_des = np.zeros_like(self._q)
        self._qddot_des = np.zeros_like(self._q)
        
        
        print(f"controller init q: {self._q.T}")



        ### custom 2 : controller 관련 
        # ================= Peg-in-hole PSFT 초기화 =================
        # 시뮬레이션 시간 변수

        

        
        self.time = 0.0



        target_SE3 = np.array([
            [ 0.9918 ,-0.1278 ,-0.0075  ,0.5],
            [-0.1278, -0.9918, -0.0014,  0.0001],
            [ -0.0073,  0.0023, -1. ,     0.25],
            [ 0.0000,  0.0000,  0.0000,  1.0000]
        ])




        T_init = target_SE3
        p_init = T_init[0:3, 3]
        R_init = T_init[0:3, 0:3]

        # contact frame은 일단 EE frame과 같다고 가정 (나중에 hole frame으로 교체 가능)
        p_contact_center_world = p_init
        R_contact_world = R_init

        # 원하는 peg 자세: EE 기준에서 살짝 tilt (y축 기준 5도 회전 예시)
        tilt = np.deg2rad(5.0)
        R_tilt_y = np.array([
            [ np.cos(tilt), 0, np.sin(tilt)],
            [ 0,            1, 0           ],
            [-np.sin(tilt), 0, np.cos(tilt)],
        ])
        R_des_world = R_contact_world @ R_tilt_y

        # PSFT 생성기
        self.psft_generator = PegInHolePSFT(
            p_contact_center_world=p_contact_center_world,
            R_contact_world=R_contact_world,
            R_des_world=R_des_world,
            f_axial_mag=3.0,            # 일단 힘도 살살 (3N 정도)
            r_min=0.0005,
            r_max=0.0030,
            theta_max=np.deg2rad(30.0),
            h=0.0001,
            dt=self.dt,
            direction='ccw',
            z_offset=0.0,
        )
        # ==============================



    def _get_robot_states(self):

        for i, idx in enumerate(self.RobotMovableJointIdx):
            states = p.getJointState(self.robotId, idx, physicsClientId=self.ClientId)

            self._q[i, 0] = states[0]     # q
            self._qdot[i, 0] = states[1]  # qdot
            self._qddot[i, 0] = 0         # TODO

        for idx in self.RobotFTJointIdx:
            states = p.getJointState(self.robotId, idx, physicsClientId=self.ClientId)
            for i in range(6):
                self._ft[i, 0] = -states[2][i]  # j

        self._ft_filtered = (1 - self._alpha) * self._ft_filtered + self._alpha * self._ft
        self._ft = self._ft_filtered
        
        self._T_end = self.pinModel.FK(self._q)

        self._Js = self.pinModel.Js(self._q)
        self._Jb = self.pinModel.Jb(self._q)
        self._Jsinv = np.linalg.pinv(self._Js)
        self._Jbinv = np.linalg.pinv(self._Jb)

        self._M = self.pinModel.M(self._q)
        self._C = self.pinModel.C(self._q, self.qdot)
        self._c = self._C @ self._qdot
        self._g = self.pinModel.g(self._q)

        self._p = SE32PoseVec(self._T_end)

        R_end = self._T_end[0:3, 0:3]
        A_upper = np.concatenate((np.zeros([3, 3]), R_end), axis=1)
        A_lower = np.concatenate((np.eye(3), np.zeros([3, 3])), axis=1)
        A = np.concatenate((A_upper, A_lower), axis=0)

        self._Jr = A @ self._Jb
        self._Jrinv = np.linalg.pinv(self._Jr)

        p.resetBasePositionAndOrientation(bodyUniqueId=self._endID, posObj=self._p[0:3, 0],
                                          ornObj=Rot2quat(self._T_end[0:3, 0:3]), physicsClientId=self.ClientId)



    def set_control_mode(self, mode: str):
        # pb.my_robot.set_control_mode("idc") 


        allowed = ["idc", "imp", "peg_imp", "pidc"]
        if mode not in allowed:
            raise ValueError(f"Unknown control mode: {mode}, allowed: {allowed}")
        print(f"[Controller] Switching mode -> {mode}")
        self.control_mode = mode

        if mode == "peg_imp":
            # 예: PSFT 리셋
            if hasattr(self, "psft_generator"):
                self.psft_generator.reset()


    def _compute_torque_input(self):
        self.time += self.dt

        if self.control_mode == "idc":
            self._compute_torque_input_idc()

        elif self.control_mode == "imp":
            self._compute_torque_input_impedance()

        elif self.control_mode == "peg_imp":
            self._compute_torque_input_peg_impedance()

        elif self.control_mode == "pidc":
            self._compute_torque_input_pidc()

        else:

            self._tau = self._g




    ## idc
    def _compute_torque_input_idc(self):

        # You need to implement robot controllers here!
        if True:
                Kp = 4000
                Kd = 200

                qddot = self._qddot_des + Kp * (self._q_des - self._q) + Kd * (self._qdot_des - self._qdot)

                tau = self._M @ qddot + self._c + self._g

                self._tau = tau

        else:
            self._tau = self._g


            

    ## imp
    def _compute_torque_input_impedance(self):
        if (not self._is_ft_calibrated) and (self.time > 1.0):
            self._ft_bias = self._ft.copy()
            self._is_ft_calibrated = True
            print(f"[PybulletRobot] F/T Sensor Calibrated. Bias: {self._ft_bias.flatten()}")

        if self._is_ft_calibrated:
            wrench_ext_body = self._ft - self._ft_bias   # (6,1)
        else:
            wrench_ext_body = np.zeros((6, 1))


        tau_ext = self._Jb.T @ wrench_ext_body          # (N,1)
        #tau_ext = 0
        tau_ext = np.clip(tau_ext, -10.0, 10.0)         

        
        if getattr(self, "enable_force_control", False):
            Fz_des = float(getattr(self, "desired_normal_force", 5.0))
        else:
            Fz_des = 0.0

        wrench_des_body = np.zeros((6, 1))
        wrench_des_body[2, 0] = Fz_des

        tau_des = self._Jb.T @ wrench_des_body          # (N,1)

        
        e     = self._q_des    - self._q                # (N,1)
        e_dot = self._qdot_des - self._qdot             # (N,1)

      
        K_val = 300                                  # stiffness (scalar → element-wise)
        K = K_val * np.ones_like(self._q)               # (N,1)

        M_diag = np.diag(self._M).reshape(-1, 1)        # (N,1)
        M_diag = np.clip(M_diag, 0.1, np.inf)
        Lambda = 0.4* M_diag                                 # (N,1)

        D = 1.0 * np.sqrt(K * Lambda)                   # (N,1)

        eps = tau_des - tau_ext                         # (N,1)

        qddot_cmd = self._qddot_des + (D * e_dot + K * e - eps) / Lambda   # (N,1)


        tau = self._M @ qddot_cmd + self._c + self._g - tau_ext           # (N,1)

        self._tau = tau

        

    ## pidc
    def _compute_torque_input_pidc(self):
        # Passivity-based Inverse Dynamics Control (P-IDC) Implementation
        
        # 1. 제어 게인 설정 (
        if True :
            Kp   = np.array([[150.0],[150.0],[150.0],[150.0],[100.0],[50.0],[10.0]])
            #Kd   = np.array([[20.0 ],[20.0 ],[20.0 ],[20.0 ],[15.0 ],[10.0],[5.0 ]])
            K_tau = np.array([[20.0 ],[20.0 ],[20.0 ],[20.0 ],[10.0 ],[5.0 ],[1.0 ]])

            # Kp   = 150
            Kd   = 100
            # K_tau = 20

            Ki = 0.1*Kp

            q_err = self._q_des - self._q
            dq_err = self._qdot_des - self._qdot
            
            self._iq_err += q_err * self.dt 

            ddq_ref = self._qddot_des + Kd * dq_err + Kp * q_err
            
            dq_ref = self._qdot_des + Kd * q_err + Kp * self._iq_err
            #dq_ref = self._qdot_des + Kd * q_err + Ki * self._iq_err

            tau_ref = K_tau * (dq_ref - self._qdot)

            C_term = self._C @ dq_ref 
            #C_term = self._C @ self._qdot
            
            
            
            tau = self._M @ ddq_ref + C_term + self._g + tau_ref

            #max_tau = np.array(self._jointforce).reshape(self.numJoints, 1)  # jointInfo[11] * safety_factor
            #tau = np.clip(tau, -max_tau, max_tau)

            self._tau = tau
        else:
             self._tau = self._g



    ###  Force PI Controller





    ### Force Hybrid Motion-Controller




    # ### Task Impedance 
    # def _compute_torque_input_task_impedance(self):


    # ### 논문 제어기 test
    # def _compute_torque_input_peg_impdeance(self):
    #     p_des, R_des, f_axial_wolrd = self.psft_generator.get_target(self.time)


    def target_update(self, target_pose):
        val = np.array(target_pose).flatten()
    
        # 위치 (x, y, z)
        self._p_des[0:3] = val[0:3].reshape(3, 1)
        
        # 회전 벡터 (rx, ry, rz)
        self._R_des[0:3] = val[3:6].reshape(3, 1)




    ##peg_imp
    ### 논문 제어기
    def _compute_torque_input_peg_impedance(self):

        #x_curr, R_curr, J, g,          # 로봇 현재 상태 (위치, 회전, 자코비안, 중력)
        #x_des, R_des,                  # 목표 상태 (위치, 회전)
        # --- 1. 튜닝 게인 및 설정 ---
        kp = 300.0   # Stiffness (N/m)
        kw = 3.0     # Orientation Stiffness (Nm/rad)
        f_z_des = 0 # Z축 조립 힘 (N, 아래로 누르는 힘)

        # --- 2. 현재 상태 가져오기 ---
        # 로봇의 상태는 벡터로 오므로 편의를 위해 분리
        p_curr_vec = np.array(self._p).flatten()
        x_curr = p_curr_vec[0:3]     # 현재 위치 (World)
        w_curr = p_curr_vec[3:6]     # 현재 회전 벡터
        R_curr_mat = Vec2Rot(w_curr) # 현재 회전 행렬 (Body -> World)

        # --- 3. 목표 상태 가져오기 ---
        x_des = np.array(self._p_des).flatten() # 목표 위치 (World)
        w_des = np.array(self._R_des).flatten() # 목표 회전 벡터
        R_des_mat = Vec2Rot(w_des)              # 목표 회전 행렬

        # --- 4. 힘 (Force) 계산 [World Frame] ---
        # 위치 오차 계산
        p_err = x_des - x_curr
        
        # 임피던스 힘 계산 (f_t = kp * error)
        f_t = kp * p_err

        # Selection Matrix 적용 (수동)
        # X, Y축: 위치 제어 (임피던스)
        # Z축: 힘 제어 (어셈블리 포스)
        f_cmd_world = np.zeros(3)
        f_cmd_world[0] = f_t[0]
        f_cmd_world[1] = f_t[1]
        f_cmd_world[2] = f_z_des 

        # --- 5. 모멘트 (Moment) 계산 [World Frame] ---
        # 회전 오차 행렬 (R_err = R_des * R_curr^T) -> World Frame 기준 오차
        R_err_mat = R_des_mat @ R_curr_mat.T
        
        # 회전 벡터로 변환 (Rotation Vector)
        ori_err_world = Rot2Vec(R_err_mat).flatten()
        
        # 모멘트 생성 (m = kw * error)
        m_cmd_world = kw * ori_err_world

        # --- 6. 좌표계 변환 [World -> Body] (핵심 수정 부분) ---
        # 현재 _Jb(Body Jacobian)를 사용하고 있으므로, 
        # World Frame의 힘/모멘트를 Body Frame으로 회전시켜야 함.
        
        # f_body = R_curr^T * f_world
        f_cmd_body = R_curr_mat.T @ f_cmd_world
        
        # m_body = R_curr^T * m_world
        m_cmd_body = R_curr_mat.T @ m_cmd_world

        # 6D Wrench 벡터 합치기 (Body Frame)
        f_star_body = np.hstack([f_cmd_body, m_cmd_body])

        # --- 7. 토크 계산 ---
        # tau = J^T * f_star + g
        # 이제 Jb.T (Body Jacobian Transpose)와 f_star_body (Body Wrench)의 좌표계가 일치함
        #tau = self._Jb.T @ f_star_body + self._g.flatten()
        # tau = self._g

        self._tau = self._Jb.T @ f_star_body + self._g.flatten()



    # def _compute_torque_input_peg_impedance(self):

    #     #is_searching = getattr(self, "peg_insertion_active", False)
    #     is_searching = True
    #     if is_searching:

    #         p_des, R_des, f_axial_world = self.psft_generator.get_target(self.time)
            
    #         kp_val = 300.0  
    #         kr_val = 10.0
    #         kv_val = 30.0 
            

    #         f_cmd = f_axial_world 

    #     else:
    #         if not hasattr(self, '_p_target_move'):
    #             self._p_target_move = self._T_end[0:3, 3].copy()
    #             self._R_target_move = self._T_end[0:3, 0:3].copy()
                
    #         p_des = self._p_target_move
    #         R_des = self._R_target_move
            
    #         kp_val = 2000.0 
    #         kr_val = 100.0
    #         kv_val = 60.0   #

    #         f_cmd = np.zeros(3)


    #     self.peg_in_hole_impedance_torque(
    #         p_des, R_des,
    #         f_axial_world=f_cmd,
    #         Kp_lin=np.diag([kp_val] * 3),
    #         Kp_rot=np.diag([kr_val] * 3),
    #         Kv_lin=np.diag([kv_val] * 3),
    #         Kv_rot=np.diag([1.0, 1.0, 1.0]), # 회전 댐핑은 적당히 고정
    #         add_gravity=True
    #     )

    # def orientation_error(self, R_des, R_cur):
    #     R_err = R_des @ R_cur.T 
    #     phi = pin.log3(R_err)
    #     return phi

    # def peg_in_hole_impedance_torque(
    #     self,
    #     p_des, R_des,
    #     f_axial_world,
    #     Kp_lin=np.diag([300.0, 300.0, 300.0]), # 강성 (Stiffness)
    #     Kp_rot=np.diag([10.0, 10.0, 10.0]),    # 회전 강성
    #     Kv_lin=np.diag([20.0, 20.0, 20.0]),    # 선형 댐핑 (Damping)
    #     Kv_rot=np.diag([1.0, 1.0, 1.0]),       # 회전 댐핑
    #     Omega_f=np.diag([1.0, 1.0, 1.0]),
    #     Omega_m=np.diag([1.0, 1.0, 1.0]),
    #     add_gravity=True,
    # ):
    #     # 1) 현재 q, qdot
    #     q    = self._q.reshape(-1)
    #     qdot = self._qdot.reshape(-1)

    #     model = self.pinModel.pinModel
    #     data  = self.pinModel.pinData 

    #     # 2) Pinocchio kinematics / Jacobian
    #     pin.forwardKinematics(model, data, q)
    #     pin.updateFramePlacements(model, data)
    #     pin.computeJointJacobians(model, data, q)

    #     # 현재 EE 상태 (World Frame)
    #     oMf = data.oMf[self._ee_frame_id]
    #     p_cur = oMf.translation.copy()
    #     R_cur = oMf.rotation.copy()

    #     # [중요] 현재 Cartesian 속도 계산 (v = J * qdot)
    #     # LOCAL_WORLD_ALIGNED를 썼으므로 v_cur도 [Linear_World, Angular_World] 입니다.
    #     J6 = pin.getFrameJacobian(
    #         model, data, self._ee_frame_id,
    #         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    #     )
    #     v_cartesian = J6 @ qdot
    #     v_lin = v_cartesian[0:3] # 선형 속도
    #     v_rot = v_cartesian[3:6] # 각속도

    #     # 3) 에러 계산
    #     e_p = p_des - p_cur                          
    #     phi = self.orientation_error(R_des, R_cur)   

    #     # 4) 임피던스 힘/모멘트 (Spring + Damping)
    #     # F = Kp * error - Kv * velocity
    #     f_spring = Kp_lin @ e_p
    #     f_damp   = -Kv_lin @ v_lin  # [핵심] 댐핑이 있어야 발산 안함!
        
    #     m_spring = Kp_rot @ phi
    #     m_damp   = -Kv_rot @ v_rot  # [핵심]

    #     # 합치기
    #     f_t = f_spring + f_damp
    #     m_t = m_spring + m_damp

    #     # 5) PSFT 축 방향 힘 더하고 task selection 적용
    #     f = Omega_f @ (f_t + f_axial_world)
    #     m = Omega_m @ m_t

    #     # 6) 6D wrench (world 기준)
    #     wrench = np.concatenate([f, m])

    #     # 7) Torque 변환
    #     tau_task = J6.T @ wrench

    #     # 8) 중력 보상 포함
    #     tau = tau_task.reshape(-1, 1)
    #     if add_gravity:
    #         tau += self._g

    #     self._tau = tau
    #     return tau





   
 





    

    def _control_robot(self):

        p.setJointMotorControlArray(bodyUniqueId=self.robotId, jointIndices=self.RobotMovableJointIdx,
                                    controlMode=p.TORQUE_CONTROL, forces=self._tau.reshape([self.numJoints]),
                                    physicsClientId=self.ClientId)

    @property
    def q(self):
        """
        :return: robot's current joint poses (rad)
        :rtype: np.ndarray (n-by-1)
        """
        return self._q.copy()

    @property
    def qdot(self):
        """
        :return: robot's current joint velocities (rad/s)
        :rtype: np.ndarray (n-by-1)
        """
        return self._qdot.copy()

    @property
    def qddot(self):
        """
        :return: [NOT WORK!!!] robot's current joint accelerations (rad/s^2)
        :rtype: np.ndarray (n-by-1)
        """
        return self._qddot.copy()

    @property
    def q_des(self):
        """
        :return: robot's desired joint poses (rad)
        :rtype: np.ndarray (n-by-1)
        """
        return self._q_des.copy()

    @property
    def p(self):
        """
        :return: robot's current task pose (xyz, xi)
        :rtype: np.ndarray (6-by-1)
        """
        return self._p.copy()

    @property
    def T_end(self):
        """
        :return: robot's current task pose (SE3)
        :rtype: np.ndarray (4-by-4)
        """
        return self._T_end.copy()

    @property
    def tau(self):
        """
        :return: robot's current input torques (Nm)
        :rtype: np.ndarray (n-by-1)
        """
        return self._tau.copy()

    @property
    def ft(self):
        """
        :return: F/T-sensor value
        :rtype: np.ndarray (6-by-1)
        """
        return self._ft.copy()

    @property
    def Js(self):
        """
        :return: Spatial jacobian in robot's current configuration
        :rtype: np.ndarray (6-by-n)
        """
        return self._Js.copy()

    @property
    def Jb(self):
        """
        :return: Body jacobian in robot's current configuration
        :rtype: np.ndarray (6-by-n)
        """
        return self._Jb.copy()

    @property
    def Jr(self):
        """
        :return: Jacobian in robot's current configuration
        :rtype: np.ndarray (6-by-n)
        """
        return self._Jr.copy()

    def JsInv(self):
        """
        :return: Inverse of the spatial jacobian in robot's current configuration
        :rtype: np.ndarray (n-by-6)
        """
        return self._Jsinv.copy()

    def Jbinv(self):
        """
        :return: Inverse of the body jacobian in robot's current configuration
        :rtype: np.ndarray (n-by-6)
        """
        return self._Jbinv.copy()

    @property
    def Jrinv(self):
        """
        :return: Inverse of the jacobian in robot's current configuration
        :rtype: np.ndarray (6-by-n)
        """
        return self._Jrinv.copy()

    @property
    def M(self):
        """
        :return: Mass matrix of the robot in robot's current configuration
        :rtype: np.ndarray (n-by-n)
        """
        return self._M.copy()

    @property
    def C(self):
        """
        :return: Coriolis matrix of the robot in robot's current configuration
        :rtype: np.ndarray (n-by-n)
        """
        return self._C.copy()

    @property
    def c(self):
        """
        :return: Coriolis vector of the robot in robot's current configuration
        :rtype: np.ndarray (n-by-1)
        """
        return self._c.copy()

    @property
    def g(self):
        """
        :return: Gravity vector of the robot in robot's current configuration
        :rtype: np.ndarray (n-by-1)
        """
        return self._g.copy()

    @property
    def q_lower(self):
        """
        :return: Lower joint pose limits (rad)
        :rtype: np.ndarray (n-by-1)
        """
        return self._jointpos_lower.copy()

    @property
    def q_upper(self):
        """
        :return: Upper joint pose limits (rad)
        :rtype: np.ndarray (n-by-1)
        """
        return self._jointpos_upper.copy()

    # Constraints visualization utils
    def _constraint_check(self):
        """
        This method provide functions for check constraints.
        It can check three kinds of limits: (i) joint's position limits, (ii) joint's velocity limits, (iii) and collision.
        If constraints are not kept, the method will change the value _xxx_flag of each joint.
        The value _xxx_flag will be used in _constraint_visualizer method to change the robot bodies' color.

        0: false -> false
        1: true -> false
        2: false -> true
        3: true -> true
        """

        # Joint position limit check
        for idx in range(self.numJoints):

            q = self._q[idx, 0] # current position of joint
            ql = self._jointpos_lower[idx]
            qu = self._jointpos_upper[idx]

            if q < ql or q > qu:
                if self._jointpos_flag[idx] == 0 or self._jointpos_flag[idx] == 1:
                    self._jointpos_flag[idx] = 2
                else:
                    self._jointpos_flag[idx] = 3
            else:
                if self._jointpos_flag[idx] == 2 or self._jointpos_flag[idx] == 3:
                    self._jointpos_flag[idx] = 0
                else:
                    self._jointpos_flag[idx] = 1

        # Joint velocity limit check
        for idx in range(self.numJoints):

            qdot = self._qdot[idx, 0]

            if np.abs(qdot) > self._jointvel[idx]:
                if self._jointvel_flag[idx] == 0 or self._jointvel_flag[idx] == 1:
                    self._jointvel_flag[idx] = 2
                else:
                    self._jointvel_flag[idx] = 3
            else:
                if self._jointvel_flag[idx] == 2 or self._jointvel_flag[idx] == 3:
                    self._jointvel_flag[idx] = 0
                else:
                    self._jointvel_flag[idx] = 1

        # Collision check
        for idx in range(self.numBodies):
            _contact_points_info = p.getContactPoints(bodyA=self.robotId, linkIndexA=idx-1,
                                                        physicsClientId=self.ClientId)
            if len(_contact_points_info) != 0:
                if self._collision_flag[idx] == 0 or self._collision_flag[idx] == 1:
                    self._collision_flag[idx] = 2
                else:
                    self._collision_flag[idx] = 3
            else:
                if self._collision_flag[idx] == 2 or self._collision_flag[idx] == 3:
                    self._collision_flag[idx] = 0
                else:
                    self._collision_flag[idx] = 1

    def _constraint_visualizer(self):
        """
        This method can visualize whether the robot violates the constraints.
        If the robot violates the constraints, its color will be changed.
        """
        self._constraint_check()

        # Joint position limit check
        for i, idx in enumerate(self.RobotMovableJointIdx):
            if self._jointpos_flag[i] == 2:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx, rgbaColor=[0, 0.7, 0, 1],
                                    physicsClientId=self.ClientId)
            elif self._jointpos_flag[i] == 0:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx, rgbaColor=self._robot_color[idx],
                                    physicsClientId=self.ClientId)
                
        # Joint velocity limit check
        for i, idx in enumerate(self.RobotMovableJointIdx):

            if self._jointvel_flag[i] == 2:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx, rgbaColor=[0, 0, 0.7, 1],
                                    physicsClientId=self.ClientId)
            elif self._jointvel_flag[i] == 0:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx, rgbaColor=self._robot_color[idx],
                                    physicsClientId=self.ClientId)
                
        # Collision check
        for idx in range(self.numBodies):

            if self._collision_flag[idx] == 2:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx-1, rgbaColor=[0.7, 0, 0, 1],
                                    physicsClientId=self.ClientId)
            elif self._collision_flag[idx] == 0:
                p.changeVisualShape(objectUniqueId=self.robotId, linkIndex=idx-1, rgbaColor=self._robot_color[idx],
                                    physicsClientId=self.ClientId)

    # Kinematics utils
    # def spatial_jacobian(self, q):
    #     # Implement this function!
    #     """
    #     :param np.ndarray q: given robot joint position (rad)
    #
    #     :return: spatial jacobian in given configuration
    #     :rtype: np.ndarray (6-by-n)
    #     """
    #
    #     Js = np.zeros([6, self.numJoints])
    #     return Js
    #
    # def body_jacobian(self, q):
    #     # Implement this function!
    #     """
    #     :param np.ndarray q: given robot joint position (rad)
    #
    #     :return: body jacobian in given configuration
    #     :rtype: np.ndarray (6-by-n)
    #     """
    #
    #     Jb = np.zeros([6, self.numJoints])
    #     return Jb

    def jacobian(self, q):
        # Implement this function!
        """
        :param np.ndarray q: given robot joint position (rad)

        :return: jacobian in given configuration
        :rtype: np.ndarray (6-by-n)
        """

        Jr = np.zeros([6, self.numJoints])
        return Jr

    def forward_kinematics(self, theta):
        # Implement this function!
        """
        :param np.ndarray q: given robot joint position (rad)

        :return: task pose corresponded to the given joint position (SE3)
        :rtype: np.ndarray (4-by-4)
        """

        T = np.eye(4)
        theta = np.asarray(theta).reshape(-1)
        n = self.numJoints
        xi_list, q0_list, g0, end_off = self.xi_list(n)

        for xi_i,q_i, th_i in zip(xi_list, q0_list, theta):
            T = T @ self.exp_twist(xi_i, th_i, q_i)

        return T @ g0 @ end_off
    


    def exp_twist(self, xi, theta, q_i):
        xi = np.asarray(xi).reshape(6)
        v, w = xi[:3], xi[3:]
        w_norm = np.linalg.norm(w)

        g = np.eye(4)
        if w_norm < 1e-12: 
            g[:3,:3] = np.eye(3)
            g[:3, 3] = v * theta
            return g

        w = w / w_norm
        R = np.eye(3) + np.sin(theta)*self.skew(w) + (1 - np.cos(theta))*(self.skew(w) @ self.skew(w))
        p = (np.eye(3)-R) @ q_i

        g[:3,:3] = R
        g[:3, 3] = p.reshape(3)
        return g
    



    

    def xi_list(self, dof):
        # xi(v,w) -> v = -wxq
        if(dof == 6):
            w = [
                np.array([[0], [0], [1]]),      
                np.array([[0], [-1], [0]]), 
                np.array([[0], [-1], [0]]), 
                np.array([[0], [0], [1]]), 
                np.array([[0], [-1], [0]]),
                np.array([[0], [0], [1]])  
            ]
            
            q0 = [
                np.array([[0], [0], [0.0775]]),      
                np.array([[0], [-0.109], [0.2995]]), 
                np.array([[0], [-0.0785], [0.7495]]), 
                np.array([[0], [-0.0035], [1.0165]]), 
                np.array([[0], [-0.1175], [1.0995]]),
                np.array([[0], [-0.2025], [1.0995]])  
            ]

            xi = [self.cal_xi_i(w_i, q_i) for w_i, q_i in zip(w, q0)]

            end_off = np.array([
                [1, 0, 0, 0],   
                [0, 1, 0, 0],
                [0, 0, 1, 0.112],
                [0, 0, 0, 1.0]     
            ], dtype=float)

            g0 = np.array([
                [1, 0, 0, 0],   
                [0, 1, 0, -0.2025],
                [0, 0, 1, 1.0995],
                [0, 0, 0, 1.0]     
            ], dtype=float)
        

        elif(dof == 7):
            w = [
                np.array([[0], [0], [1]]),      
                np.array([[0], [1], [0]]), 
                np.array([[0], [0], [1]]), 
                np.array([[0], [-1], [0]]), 
                np.array([[0], [0], [1]]),
                np.array([[0], [-1], [0]]),
                np.array([[0], [0], [-1]])    
            ]

            q0 = [
                np.array([[0], [0], [0.3330]]),      
                np.array([[0], [0], [0.3330]]), 
                np.array([[0], [0], [0.6490]]), 
                np.array([[0.0825], [0], [0.6490]]), 
                np.array([[0], [0], [1.0330]]),
                np.array([[0], [0], [1.0330]]),
                np.array([[0.0880], [0], [1.0330]])    
            ]

            end_off = np.array([
                [1, 0, 0, 0],   
                [0, 1, 0, 0],
                [0, 0, 1, 0.1070],
                [0, 0, 0, 1.0]     
            ], dtype=float)


            g0 = np.array([
                [1, 0, 0, 0.0880],   
                [0, -1, 0, 0],
                [0, 0, -1, 1.0330],
                [0, 0, 0, 1.0]     
            ], dtype=float)


            xi = [self.cal_xi_i(w_i, q_i) for w_i, q_i in zip(w, q0)]

        return xi, q0, g0, end_off
    


    def cal_xi_i(self, w_i, q_i):
        w_i = np.asarray(w_i).reshape(3)   
        q_i = np.asarray(q_i).reshape(3)
        v_i = -np.cross(w_i, q_i)        
        xi_i = np.concatenate([v_i, w_i]).reshape(6, 1)  
        return xi_i

    def skew(self,w):
        wx, wy, wz = np.asarray(w).reshape(3)
        return np.array([[0,   -wz,  wy],
                        [wz,   0 , -wx],
                        [-wy,  wx,  0 ]], dtype=float)

    def inverse_kinematics(self, T):
        # Implement this function!
        """
        :param np.ndarray T: given robot target pose in SE3

        :return: joint pos corresponded to the given task pose (rad)
        :rtype: np.ndarray (n-by-1)
        """
        q = np.zeros([self.numJoints, 1])
        return q

    # control utils
    def set_desired_joint_pos(self, q_des):
        q_des = np.asarray(q_des).reshape(-1, 1)
        self._q_des = q_des






class PSFTGenerator:
    """
    2D partial spiral force trajectory (contact plane 상에서 r, theta 생성).
    논문의 Eq. (10)에 해당하는 부분을 구현.
    """

    def __init__(self, r_min, r_max, theta_max, h, dt=0.001, direction='ccw'):
        assert r_max > r_min, "r_max must be larger than r_min"
        assert theta_max > 0.0, "theta_max must be positive"

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_max = float(theta_max)
        self.h = float(h)
        self.dt = float(dt)

        self.dir_sign = 1.0 if direction == 'ccw' else -1.0

        # 내부 상태
        self.theta = 0.0          # 현재 각도 theta_t (부호 있음)
        self.theta_hat = 0.0      # 누적 각도 \hat{theta}_t (항상 양수)
        self.r = self.r_min       # 현재 반경 r_t

    @staticmethod
    def compute_theta_max(e_r, l, beta):
        """
        논문 Eq. (9):

            theta_max > 2 * atan( 2 * e_r / (l + l * cos(beta)) )
        """
        return 2.0 * np.arctan2(2.0 * e_r, (l + l * np.cos(beta)))

    def reset(self, theta0=0.0, r0=None, direction=None):
        self.theta = float(np.clip(theta0, -self.theta_max, self.theta_max))
        self.r = float(self.r_min if r0 is None else np.clip(r0,
                                                             self.r_min,
                                                             self.r_max))
        self.theta_hat = 0.0
        if direction is not None:
            self.dir_sign = 1.0 if direction == 'ccw' else -1.0

    def step(self):
        """
        한 스텝 진행시키고 2D center point p_t 반환.
        Returns
        -------
        p_t : (2,) ndarray
        r_t : float
        theta_t : float
        """
        # dtheta = atan(h / r_t)
        dtheta = np.arctan2(self.h, max(self.r, 1e-6))

        # \hat{theta}_t 업데이트 (절댓값 누적)
        self.theta_hat += abs(dtheta)

        # r_t = r_min + \hat{theta}_t * d_r / (2 * theta_max)
        d_r = self.r_max - self.r_min
        self.r = self.r_min + (self.theta_hat * d_r) / (2.0 * self.theta_max)
        if self.r > self.r_max:
            self.r = self.r_max

        # 부호 있는 theta 업데이트
        self.theta += self.dir_sign * dtheta

        # partial spiral 범위 넘어가면 방향 반전
        if self.theta > self.theta_max:
            self.theta = self.theta_max
            self.dir_sign = -1.0
        elif self.theta < -self.theta_max:
            self.theta = -self.theta_max
            self.dir_sign = +1.0

        p_t = np.array([
            self.r * np.cos(self.theta),
            self.r * np.sin(self.theta),
        ])

        return p_t, self.r, self.theta

    def get_state(self):
        p_t = np.array([
            self.r * np.cos(self.theta),
            self.r * np.sin(self.theta),
        ])
        return p_t, self.r, self.theta, self.theta_hat


class PegInHolePSFT:
    """
    Peg-in-hole용 고수준 PSFT 모듈.

    - 내부에서 PSFTGenerator로 contact plane 상의 2D spiral center p_t (x,y)를 생성
    - hole/contact frame -> world frame 변환해서 p_des (3D)로 반환
    - orientation은 초기 설정 R_des_world 고정(논문 tilt 자세 등)
    - 축 방향 원하는 힘 f_axial_world도 같이 반환

    get_target(t)는 지금 네 컨트롤러에서 호출하는 형태에 맞춰서
    (p_des, R_des, f_axial_world)를 돌려준다.
    """

    def __init__(
        self,
        p_contact_center_world,   # 구멍 중심 (contact plane 원점) [3]
        R_contact_world,          # contact frame의 회전 (3x3), z축: 삽입축
        R_des_world,              # 원하는 peg 자세 (3x3) – 보통 tilt가 포함된 고정자세
        f_axial_mag,              # 축 방향 원하는 힘 크기 [N]  (contact z축 방향)
        r_min=0.0005,
        r_max=0.0030,
        theta_max=np.deg2rad(30),
        h=0.0001,
        dt=0.001,
        direction='ccw',
        z_offset=0.0,
    ):
        self.p_contact_center_world = np.asarray(p_contact_center_world).reshape(3)
        self.R_contact_world = np.asarray(R_contact_world).reshape(3, 3)
        self.R_des_world = np.asarray(R_des_world).reshape(3, 3)

        self.f_axial_mag = float(f_axial_mag)
        self.z_offset = float(z_offset)

        self.psft = PSFTGenerator(
            r_min=r_min,
            r_max=r_max,
            theta_max=theta_max,
            h=h,
            dt=dt,
            direction=direction,
        )

    def reset(self):
        self.psft.reset()

    def get_target(self, t=None):
        """
        네 컨트롤러에서 호출하는 인터페이스:

            p_des, R_des, f_axial_world = psft_generator.get_target(self.time)

        Parameters
        ----------
        t : float or None
            시뮬 시간. 현재 구현에서는 사용하지 않고 내부 step만 진행.
            (필요하면 여기다가 z 방향 스케줄링 넣어서 time-based insertion도 가능)

        Returns
        -------
        p_des_world : (3,) ndarray
        R_des_world : (3,3) ndarray
        f_axial_world : (3,) ndarray
        """
        # 1) contact plane 상의 2D spiral center
        p_t_2d, _, _ = self.psft.step()    # [x, y]

        # 2) contact frame에서의 3D 위치 (z=0 plane)
        p_contact = np.array([p_t_2d[0], p_t_2d[1], self.z_offset])

        # 3) world frame으로 변환
        p_des_world = self.R_contact_world @ p_contact + self.p_contact_center_world

        # 4) orientation: 초기 설정값 유지 (논문에서는 기울어진 자세 고정)
        R_des_world = self.R_des_world

        # 5) 축 방향 힘: contact frame z축 방향으로 f_axial_mag
        z_axis_world = self.R_contact_world[:, 2]  # contact frame의 z축
        f_axial_world = self.f_axial_mag * z_axis_world

        return p_des_world, R_des_world, f_axial_world