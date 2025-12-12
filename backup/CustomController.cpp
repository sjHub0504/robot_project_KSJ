//
// Created by panda on 23. 8. 17.
//

#include "CustomController.h"

void CustomController::initCustomController() {
    _initRFT44();
    _initAbstractController();
    _resetController();
    return;
}

void CustomController::resetCustomController() {
    _resetController();
}

void CustomController::_resetController() {

    _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;

    _resetRobot();
    _resetNominalRobot();

    MotionGeneratorStates._q_goal = JointVec(Robot.q);
    MotionGeneratorStates._dq_goal.setZero();
    MotionGeneratorStates._ddq_goal.setZero();

    MotionGeneratorStates._T_goal = SE3(Robot.T);
    MotionGeneratorStates._V_goal.setZero();
    MotionGeneratorStates._dV_goal.setZero();

    MotionGeneratorStates._q_des = JointVec(Robot.q);
    MotionGeneratorStates._dq_des.setZero();
    MotionGeneratorStates._ddq_des.setZero();

    MotionGeneratorStates._T_des = SE3(Robot.T);
    MotionGeneratorStates._V_des.setZero();
    MotionGeneratorStates._dV_des.setZero();

    iq_err.setZero();
    iqn_err.setZero();
    ip_err.setZero();
    ipn_err.setZero();

    return;
}


// The method to compute the next desired joint state to move toward to a joint goal state.
void CustomController::_updateDesiredJointState() {

    JointVec q_target, dq_target, ddq_target;

    JointVec q_curr = JointVec(MotionGeneratorStates._q_des);
    JointVec dq_curr = JointVec(MotionGeneratorStates._dq_des);

    JointVec q_goal = JointVec(MotionGeneratorStates._q_goal);

    switch (_tracking_mode) {
        case CustomTrackingMode::TRACKMODE_CUSTOM_JOINT_MOVING:
        {
            _computeNextJointPose(q_curr, dq_curr, q_goal, q_target, dq_target);
            _robot_state = CustomStateCode::STATE_MOVE;
        } break;
        default:
        {
            q_target = JointVec(MotionGeneratorStates._q_goal);
            dq_target = JointVec(MotionGeneratorStates._dq_goal);
            _robot_state = CustomStateCode::STATE_STOP;
        } break;
    }
    ddq_target.setZero();

    MotionGeneratorStates._q_des = JointVec(q_target);
    MotionGeneratorStates._dq_des = JointVec(dq_target);
    MotionGeneratorStates._ddq_des = JointVec(ddq_target);
}




void CustomController::_updateDesiredTaskState() {

    SE3 T_target;
    TaskVec V_target, dV_target;

    SE3 T_curr = SE3(MotionGeneratorStates._T_des);
    TaskVec V_curr = TaskVec(MotionGeneratorStates._V_des);

    SE3 T_goal = SE3(MotionGeneratorStates._T_goal);

    switch(_tracking_mode) {
        case CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING:
        {
            _computeNextTaskPose(T_curr, V_curr, T_goal, T_target, V_target);
            _robot_state = CustomStateCode::STATE_MOVE;
        }break;
        default:
        { // regulation
            T_target = SE3(MotionGeneratorStates._T_goal);
            V_target = TaskVec(MotionGeneratorStates._V_goal);
            _robot_state = CustomStateCode::STATE_STOP;
        } break;
    }

    dV_target.setZero();

    MotionGeneratorStates._T_des = SE3(T_target);
    MotionGeneratorStates._V_des = TaskVec(V_target);
    MotionGeneratorStates._dV_des = TaskVec(dV_target);
}



void CustomController::computeTorqueInput(JointVec &tau_d) {

    JointVec tau;
    JointVec tau_n;

    switch(_control_mode) {
        case CustomControlMode::CTRLMODE_CUSTOM_GRAVITY_COMPENSATION:
        {
            tau << Robot.g;
            tau_n << nomRobot.g;
            _resetNominalRobot();
        } break;
        case CustomControlMode::CTRLMODE_CUSTOM_JOINT_PIDC: // Joint Passivity-based Inverse Dynamics Control
        {
            _updateDesiredJointState(); // Used to update the next desired joint states. You can use your own trajectory instead of this method

            JointVec q_target = JointVec(MotionGeneratorStates._q_des);
            JointVec dq_target = JointVec(MotionGeneratorStates._dq_des);
            JointVec ddq_target = JointVec(MotionGeneratorStates._ddq_des);

            // Real Robot
            JointVec q_err = q_target - Robot.q;
            JointVec dq_err = dq_target - Robot.dq;
            iq_err += q_err*0.001;

            JointVec ddq_ref = ddq_target + jIDC_V*dq_err + jIDC_P*q_err;
            JointVec dq_ref = dq_target + jIDC_V*q_err + jIDC_P*iq_err;
            JointVec tau_ref = jIDC_K*(dq_ref - Robot.dq);

            tau = Robot.M * ddq_ref + Robot.C * dq_ref + Robot.g + tau_ref;

            // Nominal Robot
            JointVec qn_err = q_target - nomRobot.q;
            JointVec dqn_err = dq_target - nomRobot.dq;
            iqn_err += qn_err*0.001;

            JointVec ddqn_ref = ddq_target + jIDC_V*dqn_err + jIDC_P*qn_err;
            JointVec dqn_ref = dq_target + jIDC_V*qn_err + jIDC_P*iqn_err;
            JointVec taun_ref = jIDC_K*(dqn_ref - nomRobot.dq);

            tau_n = nomRobot.M * ddqn_ref + nomRobot.C * dqn_ref + nomRobot.g + taun_ref;

        } break;
        case CustomControlMode::CTRLMODE_CUSTOM_TASK_PIDC: // Task Passivity-based Inverse Dynamics Control
        {
            _updateDesiredTaskState();

            SE3 T_target = SE3(MotionGeneratorStates._T_des);
            TaskVec V_target = TaskVec(MotionGeneratorStates._V_des);
            TaskVec dV_target = TaskVec(MotionGeneratorStates._dV_des);

            // Real Robot
            SE3 T_tilde = TransInv(Robot.T) * T_target;
            TaskMat IAd_tilde, Ad_tilde, ad_tilde;
            Adjoint(T_tilde, Ad_tilde);
            IAdjoint(T_tilde, IAd_tilde);

            TaskVec V_tilde = V_target - IAd_tilde*Robot.V;;
            adjoint(V_tilde, ad_tilde);

            TaskVec lamb, dlamb;
            TaskMat dexp, dexpinv, ddexp;

            SE3ToVec(T_tilde, lamb);
            ip_err += lamb*0.001;
            dexp_se3(-lamb, dexp);
            dexpInv_se3(-lamb, dexpinv);
            ddexp_se3(-lamb, -dlamb, ddexp);
            dlamb = dexpinv * V_tilde;

            TaskVec ddlamb_ref = -tIDC_V*dlamb - tIDC_P*lamb;
            TaskVec dlamb_ref = -tIDC_V*lamb - tIDC_P*ip_err;

            TaskVec dV_ref = Ad_tilde * (dV_target - dexp*ddlamb_ref + ad_tilde*V_target - ddexp*dlamb);
            TaskVec V_ref = Ad_tilde * (V_target - dexp*dlamb_ref);

            Eigen::Matrix<double, 7, 6> Jinv;
            DampedJacobianInverse(Robot.J, Jinv);

            JointVec dq_ref = Jinv * (V_ref);
            JointVec ddq_ref = Jinv * (dV_ref - Robot.dJ * dq_ref);
            JointVec tau_ref = tIDC_K * (dq_ref - Robot.dq);

            tau = Robot.M * ddq_ref + Robot.C * dq_ref + Robot.g + tau_ref;

            // Nominal Robot
            SE3 T_tilde_n = TransInv(nomRobot.T) * T_target;
            TaskMat IAd_tilde_n, Ad_tilde_n, ad_tilde_n;
            Adjoint(T_tilde_n, Ad_tilde_n);
            IAdjoint(T_tilde_n, IAd_tilde_n);

            TaskVec V_tilde_n = V_target - IAd_tilde_n * nomRobot.V;;
            adjoint(V_tilde_n, ad_tilde_n);

            TaskVec lamb_n, dlamb_n;
            TaskMat dexp_n, dexpinv_n, ddexp_n;

            SE3ToVec(T_tilde_n, lamb_n);
            ipn_err += lamb_n*0.001;
            dexp_se3(-lamb_n, dexp_n);
            dexpInv_se3(-lamb_n, dexpinv_n);
            ddexp_se3(-lamb_n, -dlamb_n, ddexp_n);
            dlamb_n = dexpinv_n * V_tilde_n;

            TaskVec ddlamb_ref_n = -tIDC_V*dlamb_n - tIDC_P*lamb_n;
            TaskVec dlamb_ref_n = -tIDC_V*lamb_n - tIDC_P*ipn_err;

            TaskVec dV_ref_n = Ad_tilde_n * (dV_target - dexp_n*ddlamb_ref_n + ad_tilde_n*V_target - ddexp_n*dlamb_n);
            TaskVec V_ref_n = Ad_tilde_n * (V_target - dexp_n*dlamb_ref_n);

            Eigen::Matrix<double, 7, 6> Jinv_n;
            DampedJacobianInverse(nomRobot.J, Jinv_n);

            JointVec dq_ref_n = Jinv_n * (V_ref_n);
            JointVec ddq_ref_n = Jinv_n * (dV_ref_n - nomRobot.dJ * dq_ref_n);
            JointVec tau_ref_n = tIDC_K * (dq_ref_n - nomRobot.dq);

            tau_n = nomRobot.M* ddq_ref_n + nomRobot.C * dq_ref_n + nomRobot.g + tau_ref_n;

        } break;
        case CustomControlMode::CTRLMODE_CUSTOM_J_IMP: // Impedence
        {
            _updateDesiredJointState(); // Used to update the next desired joint states. You can use your own trajectory instead of this method

            JointVec q_target = JointVec(MotionGeneratorStates._q_des);
            JointVec dq_target = JointVec(MotionGeneratorStates._dq_des);
            JointVec ddq_target = JointVec(MotionGeneratorStates._ddq_des);

            JointVec q_err = q_target - Robot.q;
            JointVec dq_err = dq_target - Robot.dq;

            Vector6d F_ext = Robot.FT;

            JointVec tau_ext = Robot.J.transpose() * F_ext;

            double target_Fz = -5.0;
            Vector6d F_des = Vector6d::Zero(); 
            F_des(2) = target_Fz;

            JointVec tau_des = Robot.J.transpose() * F_des;

            JointVec epsilon = tau_des - tau_ext; 

            //force_ PD
            // double target_Fz = -3.0;
            // Vector6d F_des = Vector6d::Zero(); 
            // F_des(2) = target_Fz;

            // Vector6d F_err = F_des - F_ext;
            // //F_err(0) = 0;
            // //F_err(1) = 0;
            // // F_err(2) = 0;

            // static Vector6d F_err_prev = Vector6d::Zero(); 
            // double dt = 0.001; 
            // Vector6d dF_err = (F_err - F_err_prev) / dt;
            // F_err_prev = F_err; 

            // double Kp_f = 2;   
            // double Kd_f = 0; 

            // Vector6d F_cmd = F_des + (Kp_f * F_err) + (Kd_f * dF_err);

            // JointVec tau_cmd = Robot.J.transpose() * F_cmd; 
            // JointVec epsilon = tau_cmd - tau_ext; 
            

            JointVec M_diag = Robot.M.diagonal();
            JointVec Lambda = 0.8*M_diag;

            //jIMP_K
            JointVec K_diag = jIMP_K.diagonal();

            JointVec D = (K_diag.cwiseProduct(Lambda)).cwiseSqrt();

            JointVec impedance_term = D.cwiseProduct(dq_err) + K_diag.cwiseProduct(q_err) - epsilon;

            JointVec ddq_cmd = ddq_target + impedance_term.cwiseQuotient(Lambda);

            tau = Robot.M*ddq_cmd + Robot.C*Robot.dq + Robot.g - tau_ext;




            // Nominal Robot
            // JointVec qn_err = q_target - nomRobot.q;
            // JointVec dqn_err = dq_target - nomRobot.dq;
            
            // JointVec epsilon_n = JointVec::Zero();

            // JointVec Mn_diag = nomRobot.M.diagonal();
            // JointVec Lambda_n = 0.8 * Mn_diag; 

            // JointVec Dn = (K_diag.cwiseProduct(Lambda_n)).cwiseSqrt();

            // JointVec impedance_term_n = Dn.cwiseProduct(dqn_err) + K_diag.cwiseProduct(qn_err);
            // JointVec ddqn_cmd = ddq_target + impedance_term_n.cwiseQuotient(Lambda_n);
            // tau_n = nomRobot.M * ddqn_cmd + nomRobot.C * nomRobot.dq + nomRobot.g;
            tau_n = nomRobot.g;
            _resetNominalRobot();

        } break;
        case CustomControlMode::CTRLMODE_CUSTOM_PEG: 
        {
            _updateDesiredTaskState(); 

            SE3 T_target_se3 = SE3(MotionGeneratorStates._T_des); 
            Eigen::Vector3d p_target = T_target_se3.block<3,1>(0,3); 
            Eigen::Matrix3d R_target = T_target_se3.block<3,3>(0,0); 

            
            Eigen::Vector3d p_curr = Robot.T.block<3,1>(0,3);
            Eigen::Matrix3d R_curr = Robot.T.block<3,3>(0,0);

            Vector3d p_err = p_target - p_curr;
            double k_p = 50.0; 
            Vector3d f_t = k_p * p_err; 

            double k_d = 5.0; 
            Vector3d v_curr = (Robot.J * Robot.dq).head(3); 
            f_t -= k_d * v_curr;

            Vector3d f_a = Vector3d::Zero();
            // f_a(2) = -5.0; 

            Vector3d f_cmd;
            f_cmd(0) = f_t(0); 
            f_cmd(1) = f_t(1); 
            f_cmd(2) = f_a(2); 

            Eigen::Matrix3d R_err_mat = R_target * R_curr.transpose();
            Eigen::AngleAxisd angle_axis(R_err_mat);
            Vector3d ori_err = angle_axis.angle() * angle_axis.axis();

            double k_w = 3.0;
            Vector3d m_cmd = k_w * ori_err;

            Vector3d w_curr = (Robot.J * Robot.dq).tail(3);
            double k_wd = 0.1;
            m_cmd -= k_wd * w_curr;

            Vector6d f_star;
            f_star.head(3) = f_cmd;
            f_star.tail(3) = m_cmd;

            tau = Robot.J.transpose() * f_star + Robot.g;

            tau_n = nomRobot.g;
            _resetNominalRobot();

        } break;
        case CustomControlMode::CTRLMODE_CUSTOM_C_IMP: 
        {
            _updateDesiredTaskState();

            SE3 T_target = SE3(MotionGeneratorStates._T_des);
            TaskVec V_target = TaskVec(MotionGeneratorStates._V_des);
            TaskVec dV_target = TaskVec(MotionGeneratorStates._dV_des);

            TaskVec p_err;
            SE3ToVec(TransInv(Robot.T) * T_target, p_err);

            tau << Robot.J.transpose() * (tPID_P * p_err - tPID_D * Robot.J * Robot.dq) + Robot.c + Robot.g;

            TaskVec pn_err;
            SE3ToVec(TransInv(nomRobot.T) * T_target, pn_err);

            tau_n = nomRobot.J.transpose() * (tPID_P * p_err - tPID_D * nomRobot.J * nomRobot.dq) + nomRobot.c + nomRobot.g;
            
            // _updateDesiredTaskState();

            // SE3 T_target = SE3(MotionGeneratorStates._T_des);      
            // TaskVec V_target = TaskVec(MotionGeneratorStates._V_des);  
            // TaskVec dV_target = TaskVec(MotionGeneratorStates._dV_des); 
            
            // TaskVec p_err_body;
            // SE3ToVec(TransInv(Robot.T) * T_target, p_err_body);

            // TaskVec p_err_world;
            // //p_err_world.head(3) = Robot.R * p_err_body.head(3); 
            // //p_err_world.tail(3) = Robot.R * p_err_body.tail(3); 
            // Eigen::Matrix3d R_curr = Robot.T.block<3,3>(0,0);
            // p_err_world.head(3) = R_curr * p_err_body.head(3);
            // p_err_world.tail(3) = R_curr * p_err_body.tail(3);
 

            // TaskVec V_curr = Robot.J * Robot.dq;
            // TaskVec V_err = V_target - V_curr;

            // TaskVec M_task_diag;
            // M_task_diag << 2.0, 2.0, 2.0, 0.5, 0.5, 0.5; 

            // TaskVec K_task_diag;
            // K_task_diag << 50.0, 50.0, 50.0, 10.0, 10.0, 10.0;

            // TaskVec D_task_diag = 2.0 * (M_task_diag.cwiseProduct(K_task_diag)).cwiseSqrt();

            // TaskVec F_ext = Robot.FT; 
            // TaskVec F_des = TaskVec::Zero(); 
            // F_des(2) = 0.0; 
            // TaskVec F_delta = F_des - F_ext;

            // // M_d * a_cmd = F_delta + D * v_err + K * p_err
            // // -> a_cmd = a_target + M_d^-1 * (-F_delta + D * v_err + K * p_err)
            
            // TaskVec impedance_force = - F_delta 
            //                         + D_task_diag.cwiseProduct(V_err) 
            //                         + K_task_diag.cwiseProduct(p_err_world);

            // TaskVec a_cmd = dV_target + impedance_force.cwiseQuotient(M_task_diag);

            // // ddq = J_pinv * (a_cmd - dJ*dq)
            // Eigen::MatrixXd J_pinv = Robot.J.completeOrthogonalDecomposition().pseudoInverse();
            
            // TaskVec dJdq = Robot.dJ*Robot.dq; 
            
            // JointVec ddq_cmd = J_pinv * (a_cmd - dJdq);

            // // tau = M(q) * ddq_cmd + C + g - J^T * F_ext
            // tau = Robot.M * ddq_cmd + Robot.c + Robot.g - Robot.J.transpose() * F_ext;


        } break;
        default:
        {
            _control_mode = CustomControlMode::CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
            tau << Robot.g;
            tau_n << nomRobot.g;
            _resetNominalRobot();
        } break;
    }

    // DO NOT CHANGE THIS PART!
    tau_d << JointVec(tau) - Robot.g;
    updateNominalRobot(tau_n);
    tau_debug = JointVec(tau_d + Robot.g);
    return;
}

void CustomController::updateNominalRobot(JointVec tau_n) {
    JointMat Minv;
    MassInverse(nomRobot.M + 0.5*JointMat::Identity(), Minv);
    JointVec _ddq = Minv * (tau_n - nomRobot.c - nomRobot.g);
    updateNominalRobotStates(_ddq);
}

// You need to add some if-else command to process your own command.
bool CustomController::processCustomJsonRPC(const Json::Value &json_request, Json::Value &json_response) {

    int result = 0;
    int errorCode = 0;

    if (json_request["params"]["customParams"]["method"].asString() == "getCurrentState") {

        Json::Value q_response;
        Json::Value q_nom_response;
        Json::Value FT_response;
        Json::Value cur_T_response;
        Json::Value cur_p_response;
        
         for (int i=0; i<7; i++) {
            q_response[i] = Robot.q[i];
            q_nom_response[i] = nomRobot.q[i];
        }
        for (int i=0; i<6; i++) {
            FT_response[i] = Robot.FT[i];
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cur_T_response[i][j] = Robot.T(i, j);
            }
        }
        // 97mm offset
        TaskVec p;
        SE3ToPoseVec(Robot.T, p);
        for (int i=0; i<6; i++) {
            cur_p_response[i] = p[i];
        }


        
        json_response["q"] = q_response;
        json_response["q_nom"] = q_nom_response;
        json_response["FT"] = FT_response;
        json_response["T"] = cur_T_response;
        json_response["p"] = cur_p_response;
        json_response["robot_state"] = _robot_state;

        json_response["method"] = "getCurrentState";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else if (json_request["params"]["customParams"]["method"].asString() == "moveJointTo") {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();
        for (int i=0; i<7; i++) {
            MotionGeneratorStates._q_goal[i] = json_request["params"]["customParams"]["q_des"][i].asFloat();
        }
        _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_JOINT_MOVING;
        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_JOINT_PIDC;
        //_control_mode = CustomControlMode::CTRLMODE_CUSTOM_J_IMP;

        PRINT_CYAN
        debug_log; std::cout << KBLD << "Joint move to: " << KBOF << MotionGeneratorStates._q_goal.transpose() << std::endl;
        PRINT_DEFAULT

        json_response["method"] = "jointMoveTo";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else if (json_request["params"]["customParams"]["method"].asString() == "moveTaskTo") {
        if (_tracking_mode != CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING) {
            _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
            _resetController();
            _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING;
        }
        Vector6d p_des;
        for (int i=0; i<6; i++) {
            p_des[i] = json_request["params"]["customParams"]["p_des"][i].asFloat();
        }
        PoseVecToSE3(p_des, MotionGeneratorStates._T_goal);

        //_control_mode = CustomControlMode::CTRLMODE_CUSTOM_TASK_PIDC;
        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_C_IMP;

        PRINT_CYAN
        debug_log; std::cout << KBLD << "Task move to:\n" << KBOF << MotionGeneratorStates._T_goal << std::endl;
        PRINT_DEFAULT

        json_response["method"] = "taskMoveTo";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }


    // else if (json_request["params"]["customParams"]["method"].asString() == "setTaksTarget") {
    //     //MotionGeneratorStates._q_goal = json_request["params"]["customParams"]["q_des"][i].asFloat();
    //     MotionGeneratorStates._T_goal = json_request["params"]["customParams"]["T_des"];


    //     //PRINT_CYAN
    //     //debug_log; std::cout << KBLD << "task_target: " << KBOF << MotionGeneratorStates._q_goal.transpose() << std::endl;
    //     //PRINT_DEFAULT

    //     json_response["method"] = "TargetTask";
    //     json_response["errorCode"] = errorCode;
    //     json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    // }

    else if (json_request["params"]["customParams"]["method"].asString() == "customJIMP") {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();

        _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_JOINT_MOVING;
        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_J_IMP;


        json_response["method"] = "JIMPcustom";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else if (json_request["params"]["customParams"]["method"].asString() == "customCIMP") {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();

        _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING;
        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_C_IMP;


        json_response["method"] = "CIMPcustom";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else if (json_request["params"]["customParams"]["method"].asString() == "customPEG") {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();

        _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING;
        _control_mode = CustomControlMode::CTRLMODE_CUSTOM_PEG;


        json_response["method"] = "PEGcustom";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }
    
    else if (json_request["params"]["customParams"]["method"].asString() == "resetFTBias") {
        PRINT_CYAN
        debug_log; std::cout << "DEPRECATED" << std::endl;
        PRINT_DEFAULT

        _flagRFT44 = CustomRFT44Flag::SET_FT_BIAS;

        json_response["method"] = "resetFTBias";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else if (json_request["params"]["customParams"]["method"].asString() == "directTeaching") {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();

        PRINT_YELLOW
        debug_log; std::cout << "Change controller to gravity compensation" << std::endl;
        PRINT_DEFAULT

        json_response["method"] = "directTeaching";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    else {
        _control_mode = CTRLMODE_CUSTOM_GRAVITY_COMPENSATION;
        _resetController();

        PRINT_RED
        debug_log; printf("UNAVAILABLE COMMAND ID!!! (%s)\n", json_request["params"]["customParams"]["method"].asString());
        PRINT_DEFAULT

        PRINT_YELLOW
        debug_log; std::cout << "Change controller to gravity compensation" << std::endl;
        PRINT_DEFAULT

        json_response["method"] = "default";
        json_response["errorCode"] = errorCode;
        json_response["message"] = (errorCode == 0) ? "Success" : "Failed";
    }

    return true;
}

// Motion Generator

bool CustomController::_computeNextJointPose(const JointVec &q_curr, const JointVec &dq_curr, const JointVec &q_goal,
                                             JointVec& q_target, JointVec& dq_target) {

    if (_tracking_mode == CustomTrackingMode::TRACKMODE_CUSTOM_JOINT_MOVING) {

        bool isFinish = MotionGenerator.computeNextJointPose_new(q_target, dq_target, q_curr, dq_curr, q_goal);

        if (((Robot.q - q_goal).norm() < 1e-2) && (Robot.dq.norm() < 1e-2)) {
            PRINT_CYAN
            debug_log; std::cout << "[computeNextJointPose] Move finished!" << std::endl;
            PRINT_DEFAULT

            dq_target.setZero();
            _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_REGULATION;
            return true;
        }
    }
    return false;
}



bool CustomController::_computeNextTaskPose(const SE3 &T_curr, const TaskVec &V_curr, const SE3 &T_goal,
                                            SE3& T_target, TaskVec& V_target) {

    if (_tracking_mode == CustomTrackingMode::TRACKMODE_CUSTOM_TASK_MOVING) {
//        bool isFinish = MotionGenerator.computeNextTaskPose_new(T_target, V_target, T_curr, V_curr, T_goal);
        bool isFinish = MotionGenerator.computeNextTaskPoseDecoupled_new(T_target, V_target, T_curr, V_curr, T_goal);

        SE3 T_err = TransInv(Robot.T) * T_goal;
        TaskVec lamb_err;
        SE3ToVec(T_err, lamb_err);
        if ((lamb_err.norm() < 1e-1) && (Robot.V.norm() < 1e-2)) {
            PRINT_CYAN
            debug_log; std::cout << "[computeNextTaskPose] Move finished!" << std::endl;
            PRINT_DEFAULT

            V_target.setZero();
            _tracking_mode = CustomTrackingMode::TRACKMODE_CUSTOM_REGULATION;
            return true;
        }
    }
    return false;
}


// Thread function
void CustomController::_threadCustomThreadHigh() {
    return;
}

void CustomController::_threadCustomThreadMid() {
    _processRFT44();
    return;
}

void CustomController::_threadCustomThreadLow() {
    return;
}

// RFT Module
bool CustomController::_initRFT44() {
    if (rft44.Connect(tty_name)) {

        PRINT_BLUE
        debug_log; std::cout << "Serial initialization start..." << std::endl;
        PRINT_DEFAULT

        unsigned char debug[16];

        rft44.read_model_name(debug);
        PRINT_GREEN
        printf("%s[Model name]%s: %s\n", KBLD, KBOF, debug);
        PRINT_DEFAULT

        rft44.read_serial_number(debug);
        PRINT_GREEN
        printf("%s[Serial number]%s: %s\n", KBLD, KBOF, debug);
        PRINT_DEFAULT

        _flagRFT44 = CustomRFT44Flag::READ_FT;
        PRINT_BLUE
        debug_log; std::cout << "Serial initialization done\n" << std::endl;
        PRINT_DEFAULT

        return true;
    }
    else {
        PRINT_RED
        std::cout << "\n[ERROR] Serial initialization failed\n" << std::endl;
        _flagRFT44 = CustomRFT44Flag::NO_DEVICE;
        PRINT_DEFAULT
        return false;
    }
}

void CustomController::_processRFT44() {
    switch (_flagRFT44) {
        case CustomRFT44Flag::READ_FT:
        {
            float ft_array[6] = {0};
            rft44.read_ft(ft_array);
            Robot.FT << ft_array[0], ft_array[1], ft_array[2], ft_array[3], ft_array[4], ft_array[5];
        }; break;
        case CustomRFT44Flag::SET_FT_BIAS:
        {
            rft44.set_bias();
            _flagRFT44 = CustomRFT44Flag::READ_FT;
        }; break;
        default:
        {
            _flagRFT44 = CustomRFT44Flag::NO_DEVICE;
        }; break;
    }
}

