//
// Created by panda on 23. 8. 17.
//

#ifndef CUSTOM_CUSTOMCONTROLLER_H
#define CUSTOM_CUSTOMCONTROLLER_H

#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <math.h>

#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "../../abstracts/AbstractController.h"
#include "../../utils/devices/RFT44_ft/rft44_utils.h"

//#include "qpOASES.hpp"

class CustomController : public AbstractController {
public:
    CustomController() {

        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_JOINT_PD;
        _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_REGULATION;
        _robot_state = CustomStateCode::STATE_STOP;

        // Joint Inverse Dynamics Control
        jIDC_P.setZero();
        jIDC_V.setZero();
        jIDC_K.setZero();

        jIDC_P.diagonal() << 150.0, 150.0, 150.0, 150.0, 100.0, 50.0, 10.0;
        jIDC_V.diagonal() << 20.0, 20.0, 20.0, 20.0, 15.0, 10.0, 5.0;
        jIDC_K.diagonal() << 20.0, 20.0, 20.0, 20.0, 10.0, 5.0, 3.0;


        // Task Inverse Dynamics Control
        tIDC_P.setZero();
        tIDC_V.setZero();
        tIDC_K.setZero();

        tIDC_P.diagonal() << 80.0, 80.0, 80.0, 10.0, 10.0, 10.0;
        tIDC_V.diagonal() << 30.0, 30.0, 30.0, 8.0, 8.0, 8.0;
        tIDC_K.diagonal() << 20.0, 20.0, 20.0, 20.0, 10.0, 5.0, 3.0;


        // Joint Compliance Control
        jPID_P.setZero();
        jPID_I.setZero();
        jPID_D.setZero();

        jPID_P.diagonal() << 200.0, 200.0, 200.0, 200.0, 30.0, 15.0, 5.0;
        jPID_I.diagonal() << 50.0, 50.0, 50.0, 50.0, 30.0, 20.0, 10.0;
        jPID_D.diagonal() << 10.0, 10.0, 10.0, 10.0, 4.0, 2.0, 1.0;

        tPID_P.setZero();
        tPID_I.setZero();
        tPID_D.setZero();
        tPID_damping.setZero();

        tPID_P.diagonal() << 300.0, 300.0, 300.0, 80.0, 80.0, 80.0;
        tPID_I.diagonal() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        tPID_D.diagonal() << 20.0, 20.0, 20.0, 15, 15, 15;
        tPID_damping.diagonal() << 0.01, 0.01, 0.01, 0.01, 0.0025, 0.0025, 0.00125;


        //Custom Impedence
        jIMP_K.setZero();
        jIMP_K.diagonal() << 25.0, 25.0, 25.0, 25.0, 10.0, 5.0, 3.0;

    

        


        iq_err.setZero();
        iqn_err.setZero();

        ip_err.setZero();
        ipn_err.setZero();

        tau_debug.setZero();
    }

    ~CustomController() {
        rft44.Disconnect();
    }

public:
    enum CustomControlMode : int {
        CTRLMODE_CUSTOM_GRAVITY_COMPENSATION = 0,
        CTRLMODE_CUSTOM_JOINT_PD = 1,
        CTRLMODE_CUSTOM_JOINT_PIDC = 3,
        CTRLMODE_CUSTOM_TASK_PIDC = 4,
        CTRLMODE_CUSTOM_J_IMP = 22,
        CTRLMODE_CUSTOM_C_IMP = 23,
        CTRLMODE_CUSTOM_PEG = 24,
    };

    enum CustomTrackingMode : int{
        TRACKMODE_CUSTOM_REGULATION = 0,
        TRACKMODE_CUSTOM_JOINT_MOVING = 1,
        TRACKMODE_CUSTOM_JOINT_WAYPOINT = 5,
        TRACKMODE_CUSTOM_TASK_MOVING = 2,
    };

    enum CustomErrorCode : int {
        ERR_SOF_NOT_MATCHED = -1,
        ERR_COMMAND_ID_NOT_MATCHED = -2,
    };

    enum CustomStateCode : int {
        STATE_STOP = 0,
        STATE_MOVE = 1,
    };

    enum CustomRFT44Flag : int {
        NO_DEVICE = 0,
        READ_FT = 1,
        SET_FT_BIAS = 2,
    };

private:
    // Controller
    int _control_mode;
    int _tracking_mode;
    int _robot_state;

    // Joint pIDC
    JointMat jIDC_P, jIDC_V, jIDC_K;

    JointVec iq_err, iqn_err;
    TaskVec ip_err, ipn_err;

    TaskMat tIDC_P, tIDC_V;
    JointMat tIDC_K;

    // Joint Impedence
    JointMat jIMP_K;

    //Joint PID
    TaskMat jPID_P, jPID_D, jPID_I;

    // Task PID
    TaskMat tPID_P, tPID_D, tPID_I;
    JointMat tPID_damping;
    

    JointVec tau_debug;

private:
    // External FT
    RFT44 rft44;
    const char* tty_name = DEFAULT_EXTERNAL_USB_PORT;
    int _flagRFT44 = CustomRFT44Flag::NO_DEVICE;

    bool _initRFT44();
    void _processRFT44();

public:
    void initCustomController() override;
    void resetCustomController() override;
    bool processCustomJsonRPC(const Json::Value &json_request, Json::Value &json_response) override;
    void computeTorqueInput(JointVec &tau_d) override;
    void updateNominalRobot(JointVec tau_n) override;

private:
    void _updateDesiredJointState();
    void _updateDesiredTaskState();


private:
    // Motion Planner

    struct AbstractMotionGeneratorStates {
        JointVec _q_goal, _q_des;
        JointVec _dq_goal, _dq_des;
        JointVec _ddq_goal, _ddq_des;

        SE3 _T_goal, _T_des;
        TaskVec _V_goal, _V_des;
        TaskVec _dV_goal, _dV_des;
    };

    AbstractMotionGenerator MotionGenerator = AbstractMotionGenerator();
    AbstractMotionGeneratorStates MotionGeneratorStates;
    
    bool _computeNextJointPose(const JointVec& q_curr, const JointVec& dq_curr, const JointVec& q_goal, JointVec& q_target, JointVec& dq_target);
    bool _computeNextTaskPose(const SE3& T_curr, const TaskVec& V_curr, const SE3& T_goal, SE3& T_target, TaskVec& V_target);


   

private:
    void _resetController();

    void _threadCustomThreadHigh() override;
    void _threadCustomThreadMid() override;
    void _threadCustomThreadLow() override;

};

#endif //CUSTOM_CUSTOMCONTROLLER_H
