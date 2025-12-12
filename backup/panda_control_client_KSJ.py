import socket
import time
import json
import numpy as np
from threading import Lock

from abstracts.abstract_jsonrpc_client import AbstractJsonrpcClient

###############################################################################
# Custom JsonRPC Class                                                           #
###############################################################################
RES_CUSTOM_SUCCESS = 0

def custom_tcp(cmd):
    def decorate(func):
        def decorated(self, *args, **kargs):
            params = func(self, *args, **kargs)
            if 'error' in self.response:
                print("[" + params["method"] + "] " + self.response['error']["message"])
            else:
                if self.response["result"]['errorCode'] == RES_CUSTOM_SUCCESS:
                    print("Response : {}".format(self.response["result"]["response"]))
                else:
                    print("failed to " + cmd + " (error code: {})".format(self.response["result"]["errorCode"]))
                # if self.response['result']['errorCode'] == RES_CUSTOM_SUCCESS:
                #     print("Response : {}".format(self.response["result"]["response"]))
                # else:
                #     print("failed to " + cmd + " (error code: {})".format(self.response['result']["errorCode"]))
                return
        return decorated
    return decorate

def any_to_list(x:list|np.ndarray):
    return np.array(x).reshape(-1).tolist()

class JsonrpcClient(AbstractJsonrpcClient):
    def __init__(self, server_ip, debugging=False):
        super().__init__(server_ip, debugging)
        
        self.response = None

        print(self._BoldText + "Server IP:" + self._ResetText, self._server_ip)
        print(self._BoldText + "Server port:" + self._ResetText, self._server_port)

    def _prepare_command_and_parse_response(self):
        self.request = json.dumps(self.data).encode()
        self.req_size = len(self.request)
        [self.response, self.res_size] = super().communicate(self.request , self.req_size)

    def get_robot_states(self):
        
        params = {}
        params["method"] = "getCurrentState"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()

        joint_pos = self.response["q"]
        nom_joint_pos = self.response["q_nom"]
        FT = self.response["FT"]
        cur_T = np.array(self.response["T"]).reshape(4, 4)
        cur_p = self.response["p"]

        #cur_T = self.response["T"]

        # end_off = np.array([
        #     [1, 0, 0, 0],   
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0.09500],
        #     [0, 0, 0, 1.0]     
        # ], dtype=float)

        # cur_T = cur_T @ end_off


        return joint_pos, nom_joint_pos, FT, cur_T, cur_p

    def is_robot_move(self):
        params = {}
        params["method"] = "getCurrentState"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()

        if self.response['robot_state'] == 1:
            return True
        return False

    def move_joint_to(self, q_des:list|np.ndarray):
        params = {}
        params["method"] = "moveJointTo"
        params["q_des"] = any_to_list(q_des)

        # return self._communicate()
        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    
    def move_task_to(self, p_des:list|np.ndarray):
        params = {}
        params["method"] = "moveTaskTo"
        params["p_des"] = any_to_list(p_des)

        # return self._communicate()
        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    

    
    
    
    
    # def set_T_target(self, ):
    #     params = {}
    #     params["method"] = "setTaksTarget"
    #     params["T_des"] = 

    #     self.data["method"] = "processCustomJsonRPC"
    #     self.data["params"] = {"customParams": params}
    #     self._prepare_command_and_parse_response()
    #     return params
    
    
    def custom_control_J_Imp(self):
        params = {}
        params["method"] = "customJIMP"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    
    def custom_control_C_Imp(self):
        params = {}
        params["method"] = "customCIMP"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    
    def custom_control_PEG(self):
        params = {}
        params["method"] = "customPEG"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    
    

    def set_ft_bias(self):
        params = {}
        params["method"] = "resetFTBias"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params

    def change_to_direct_teaching(self):
        params = {}
        params["method"] = "directTeaching"

        self.data["method"] = "processCustomJsonRPC"
        self.data["params"] = {"customParams": params}
        self._prepare_command_and_parse_response()
        return params
    
    def get_cur_T(self):
        #fdf
        jpos, jpos_nom, FT, T = self.get_robot_states()
        cur_T = self.forward_kinematics(jpos)
        return cur_T
    




    

    def forward_kinematics(self, theta):

        T = np.eye(4)
        theta = np.asarray(theta).reshape(-1)
        xi_list, q0_list, g0, end_off = self.xi_list()

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






    def xi_list(self):
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
            [0, 0, 1, 0.1955],
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
    




