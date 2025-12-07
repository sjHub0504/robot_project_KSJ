"""
PybulletCore
~~~~~~~~~~~~~
"""

import os
import sys
import time
from math import *

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread

import pybullet as p
import pybullet_data

from src.utils import *
from src.core.pybullet_robot import PybulletRobot

class PybulletCore:
    """
    Pybullet Simulator Core Class
    """
    def __init__(self):

        np.set_printoptions(linewidth=500)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=4)

        # Simulator configuration
        self.__filepath = os.path.dirname(os.path.abspath(__file__))
        self.__cfgpath  = self.__filepath + "/../configs"
        self.__urdfpath = self.__filepath + "/../assets/urdf"

        self.startPosition = [0, 0, 0] ## base position
        self.startOrientation = [0, 0, 0] ## base orientation

        self.g_vector = np.array([0, 0, -9.81]).reshape([3, 1])

        self.dt = 1. / 1000  # Simulation Frequency 240

    def connect(self, robot_name = 'indy7_v2', joint_limit=True, constraint_visualization=True):
        """
        Connect to Pybullet GUI

        :param string robot_name: robot name want to import, defaults to 'indy7_v2'
        :param bool joint_limit: activate/deactivate the joint limit constraint, defaults to True
        :param bool constraint_visualization: activate/deactivate the constraint visualizer, defaults to True
        """

        # Open GUI
        self.ClientId = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


        # Set perspective camera
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.3, 0, 0], lineColorRGB=[1, 0, 0],
                           lineWidth=5, physicsClientId=self.ClientId)
        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.3, 0], lineColorRGB=[0, 1, 0],
                           lineWidth=5, physicsClientId=self.ClientId)
        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.3], lineColorRGB=[0, 0, 1],
                           lineWidth=5, physicsClientId=self.ClientId)

        # Set physics scene
        p.setGravity(self.g_vector[0], self.g_vector[1], self.g_vector[2]) # set gravity
        #p.setGravity(0,0,0)
        p.performCollisionDetection()
        p.setTimeStep(self.dt)

        # Add plane
        self.planeId = p.loadURDF("plane.urdf")

        # hole_urdf_path = self.__urdfpath + "/franka_panda/franka_panda_handle/project_hole.urdf"
        # hole_pos = [0.0, 1.0, 0.0]  
        # hole_orn = p.getQuaternionFromEuler([0, 0, 0])
        # self.holeId = p.loadURDF(
        #         hole_urdf_path,
        #         basePosition=hole_pos,
        #         baseOrientation=hole_orn,
        #         useFixedBase=1, 
        #     )


        # visualShapeId = p.createVisualShape(
        #     shapeType=p.GEOM_MESH,
        #     fileName=self.__urdfpath + "/franka_panda/franka_panda_handle/meshes/visual/base_link.STL",
        #     meshScale=[1, 1, 1]
        # )
        
        # collisionShapeId = p.createCollisionShape(
        #     shapeType=p.GEOM_MESH,
        #     fileName=self.__urdfpath + "/franka_panda/franka_panda_handle/meshes/collision/base_link.STL",
        #     meshScale=[1, 1, 1],
        #     flags=p.GEOM_FORCE_CONCAVE_TRIMESH  # [핵심] 오목한 형상(구멍) 유지!
        # )
        
        # self.holeId = p.createMultiBody(
        #     baseMass=0,  # 0으로 하면 고정됨 (Static)
        #     baseCollisionShapeIndex=collisionShapeId,
        #     baseVisualShapeIndex=visualShapeId,
        #     basePosition=[0.5, 0, 0.25],
        #     baseOrientation=[0, 0, 0, 1]
        # )


        # Define robot's information
        robot_info = {"robot_name":None, "robot_position":None, "robot_orientation":None, "robot_properties":{}}
        robot_info["robot_name"] = robot_name
        robot_info["robot_position"] = self.startPosition
        robot_info["robot_orientation"] = self.startOrientation
        robot_info["robot_properties"]["joint_limit"] = joint_limit
        robot_info["robot_properties"]["constraint_visualization"] = constraint_visualization

        # Import robot
        self.my_robot = PybulletRobot(ClientId=self.ClientId, robot_info=robot_info, dt=self.dt)

        # Debug Frame buffer
        self._debug_frame_buff_list = []

        # Run core thread
        self.__isSimulation = False
        self._thread = Thread(target=self._thread_main)
        self._thread.start()

        # Start simulation
        self.__isSimulation = True

    def disconnect(self):
        """
        Disconnect to Pybullet GUI
        """

        self.__isSimulation = False
        time.sleep(1)
        p.disconnect(physicsClientId=self.ClientId)
        PRINT_BLUE("Disconnect Success!")

    def _thread_main(self):
        """
        Core thread of pybullet simulation framework
        """
        while True:
            ts = time.time()
            if self.__isSimulation:
                self._thread_pre()

                self.my_robot.robot_update()
                
                self._thread_post()

                p.stepSimulation()

            tf = time.time()
            if tf-ts < self.dt:
                time.sleep(self.dt-tf+ts)

    def _thread_pre(self):
        """
        This method is called at the beginning of _thread_main.
        """
        pass

    def _thread_post(self):
        """
        This method is called at the end of _thread_main.
        """
        pass

    # Debug Frame
    # def add_debug_frames(self, Tlist):

    #     if len(Tlist) > len(self._debug_frame_buff_list):
    #         for _ in range(len(Tlist) - len(self._debug_frame_buff_list)):
    #             self._debug_frame_buff_list.append(DebugFrame(self.ClientId))
    #     elif len(Tlist) < len(self._debug_frame_buff_list):
    #         for i in range(len(self._debug_frame_buff_list) - len(Tlist)):
    #             self._debug_frame_buff_list[len(Tlist) + i].setPos([0, 0, -1], [0, 0, 0])

    #     for i, T in enumerate(Tlist):
    #         self._debug_frame_buff_list[i].setSE3(T)

    def add_debug_frames(self, Tlist, palettes=None):
        """
        Tlist: [4x4, 4x4, ...]
        palettes: 각 프레임의 (x,y,z) 축 색을 지정하는 리스트
                예) [ ((0,1,1),(1,0,1),(1,1,0)),  # 프레임0 CMY
                    ((1,0,0),(0,1,0),(0,0,1)) ]  # 프레임1 RGB
        """
        import numpy as np
        default_palettes = [
            ((1,0,0),(0,1,0),(0,0,1)),   # RGB
            ((0,1,1),(1,0,1),(1,1,0)),   # CMY
            ((1,0.5,0),(0.3,0.7,1),(0.6,0.2,0.8)),  # extra
        ]
        if palettes is None:
            palettes = default_palettes
        assert len(Tlist) > 0, "Tlist is empty"

        if len(Tlist) > len(self._debug_frame_buff_list):
            for i in range(len(self._debug_frame_buff_list), len(Tlist)):
                axis_colors = palettes[i % len(palettes)]
                self._debug_frame_buff_list.append(
                    DebugFrame(self.ClientId, axis_len=0.10, lineWidth=3, axis_colors=axis_colors)
                )
        elif len(Tlist) < len(self._debug_frame_buff_list):
            for i in range(len(Tlist), len(self._debug_frame_buff_list)):
                self._debug_frame_buff_list[i].setPos([0,0,-1],[0,0,0])

        for i, T in enumerate(Tlist):
            T = np.asarray(T)
            if T.shape != (4,4):
                raise ValueError(f"SE3 must be 4x4, got {T.shape}")
            self._debug_frame_buff_list[i].setSE3(T)

    def destroy_debug_frames(self):
        for i in range(len(self._debug_frame_buff_list)):
            self._debug_frame_buff_list[0].remove()
            self._debug_frame_buff_list.pop(0)
    
    # For jupyter notebook
    def MoveRobot(self, q, degree=True, verbose=False):
        """
        Move the robot to the given joint angle
        """

        if degree:
            q = deg2radlist(q)

        self.my_robot.set_desired_joint_pos(q)

        if (verbose == True):
            PRINT_BLUE("***** Set desired joint angle *****")
            print(np.asarray(q).reshape(-1))


# class DebugFrame:
#     def __init__(self, ClientId, lineWidth=3):

#         self.ClientId = ClientId

#         visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0.5, 0.5, 0.45])

#         self._endID = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, -1],
#                                         baseOrientation=[0, 0, 0], physicsClientId=self.ClientId)

#         self._endID_x = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.08, 0, 0], lineColorRGB=[1, 0, 0],
#                                            lineWidth=lineWidth, parentObjectUniqueId=self._endID,
#                                            physicsClientId=self.ClientId)
#         self._endID_y = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.08, 0], lineColorRGB=[0, 1, 0],
#                                            lineWidth=lineWidth, parentObjectUniqueId=self._endID,
#                                            physicsClientId=self.ClientId)
#         self._endID_z = p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.08], lineColorRGB=[0, 0, 1],
#                                            lineWidth=lineWidth, parentObjectUniqueId=self._endID,
#                                            physicsClientId=self.ClientId)

#     def setSE3(self, T):
#         p.resetBasePositionAndOrientation(bodyUniqueId=self._endID, posObj=T[0:3, 3],
#                                           ornObj=Rot2quat(T[0:3, 0:3]), physicsClientId=self.ClientId)

#     def setPos(self, pos, ori):
#         self.setSE3(xyzeul2SE3(pos, ori))

#     def remove(self):
#         p.removeBody(bodyUniqueId=self._endID, physicsClientId=self.ClientId)


class DebugFrame:
    def __init__(self, ClientId, lineWidth=5, axis_len=0.09 ,
                 axis_colors=((1,0,0.1),(0,1,0.1),(0,0,1)), 
                 sphere_rgba=(0, 0.5, 0.5, 0.25)):

        self.ClientId = ClientId
        L = axis_len
        cx, cy, cz = axis_colors

        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=sphere_rgba
        )
        self._endID = p.createMultiBody(
            baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, -1],
            baseOrientation=[0, 0, 0], physicsClientId=self.ClientId
        )

        self._endID_x = p.addUserDebugLine([0,0,0], [L,0,0], cx, lineWidth,
                         parentObjectUniqueId=self._endID, physicsClientId=self.ClientId)
        self._endID_y = p.addUserDebugLine([0,0,0], [0,L,0], cy, lineWidth,
                         parentObjectUniqueId=self._endID, physicsClientId=self.ClientId)
        self._endID_z = p.addUserDebugLine([0,0,0], [0,0,L], cz, lineWidth,
                         parentObjectUniqueId=self._endID, physicsClientId=self.ClientId)

    def setSE3(self, T):
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._endID,
            posObj=T[0:3, 3],
            ornObj=Rot2quat(T[0:3, 0:3]),
            physicsClientId=self.ClientId
        )

    def setPos(self, pos, ori):
        self.setSE3(xyzeul2SE3(pos, ori))

    def remove(self):
        p.removeBody(bodyUniqueId=self._endID, physicsClientId=self.ClientId)