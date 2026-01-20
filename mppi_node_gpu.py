#!/usr/bin/env python3
import sys
import os
import rospy
import numpy as np
import torch
from kortex_driver.msg import Base_JointSpeeds, JointSpeed, BaseCyclic_Feedback
from kortex_driver.srv import *

try:
    from mppi_solver_gpu import MPPIControllerGPU
except ImportError:
    rospy.logerr("mppi_solver_gpu not found!")
    sys.exit()

class Gen3LiteMPPINodeGPU:
    def __init__(self):
        rospy.init_node('gen3_lite_mppi_gpu_node')
        
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
        
        rospy.loginfo("🚀 Initializing Joint-Space MPPI...")
        self.mppi = MPPIControllerGPU(urdf_path)
        self.device = self.mppi.device
        
        self.q_curr_full = None
        self.is_ready = False
        
        # 토픽 설정
        self.sub_feedback = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback", 
            BaseCyclic_Feedback, 
            self.cb_feedback
        )
        self.pub_vel = rospy.Publisher(
            f"/{self.robot_name}/in/joint_velocity", 
            Base_JointSpeeds, 
            queue_size=1
        )
        
        # 서비스 연결 (에러 리셋 등)
        try:
            rospy.wait_for_service(f'/{self.robot_name}/base/clear_faults', timeout=2)
            self.clear_faults = rospy.ServiceProxy(f'/{self.robot_name}/base/clear_faults', Base_ClearFaults)
        except:
            rospy.logwarn("⚠️ Kortex services not available.")

        rospy.on_shutdown(self.stop)
        rospy.loginfo(f"✅ Ready on {self.device}!")

    def cb_feedback(self, msg):
        # Gen3 Lite는 6축
        q = [np.deg2rad(msg.actuators[i].position) for i in range(6)]
        self.q_curr_full = np.array(q + [0]*4) # 10차원 패딩 (그리퍼 등)
        self.is_ready = True

    def stop(self):
        msg = Base_JointSpeeds()
        for i in range(6):
            js = JointSpeed()
            js.joint_identifier = i
            js.value = 0.0
            js.duration = 0
            msg.joint_speeds.append(js)
        self.pub_vel.publish(msg)

    def main(self):
        # 1. GPU Warm-up (필수)
        rospy.loginfo("🔥 Warming up GPU...")
        for _ in range(20):
            self.mppi.get_optimal_command(np.zeros(6), np.array([0.5,0,0.5]), np.eye(3))
        
        rospy.loginfo("⌛ Waiting for robot feedback...")
        while not self.is_ready: rospy.sleep(0.1)
        
        # 2. 목표 설정 (현재 위치에서 Z축 +15cm)
        with torch.no_grad():
            q_t = torch.tensor(self.q_curr_full[:6], device=self.device).float().unsqueeze(0)
            tg = self.mppi.dyn.chain.forward_kinematics(q_t)
            curr_pos = tg.get_matrix()[0, :3, 3].cpu().numpy()
            
        target_pos = curr_pos.copy()
        target_pos[2] += 0.15 # 15cm 위로
        target_rot = np.eye(3) # 회전은 유지 (Identity)
        
        rospy.loginfo(f"🎯 Target set: {target_pos}")
        
        # 3. 제어 루프 (20Hz)
        hz = 20
        rate = rospy.Rate(hz)
        
        while not rospy.is_shutdown():
            if not self.is_ready: continue
            
            start_t = rospy.get_time()
            
            # [핵심] MPPI 계산 (Joint Velocity 반환)
            dq_rad = self.mppi.get_optimal_command(self.q_curr_full[:6], target_pos, target_rot)
            
            #----------!!!!!!!!안전장치!!!!!!!!!!!--------
            vel_limit = 0.01            
            dq_rad = np.clip(dq_rad, -vel_limit, vel_limit)
            dq_deg = np.rad2deg(dq_rad)
            #--------------------------------------------

            # 메시지 생성
            msg = Base_JointSpeeds()
            for i in range(6):
                js = JointSpeed()
                js.joint_identifier = i
                js.value = dq_deg[i]
                js.duration = 50 # 50ms
                msg.joint_speeds.append(js)
            self.pub_vel.publish(msg)
            
            # 거리 체크
            with torch.no_grad():
                q_tensor = torch.tensor(self.q_curr_full[:6], device=self.device).float().unsqueeze(0)
                tg = self.mppi.dyn.chain.forward_kinematics(q_tensor)
                m = tg.get_matrix()[0].cpu().numpy() # (4, 4) 행렬
                
                curr_pos = m[:3, 3]
                curr_rot = m[:3, :3]
                
                # (1) 위치 오차
                pos_err = np.linalg.norm(curr_pos - target_pos)
                
                # (2) 회전 오차 (Trace Trick: 0~3.0)
                R_diff = np.matmul(target_rot.T, curr_rot)
                rot_err = 3.0 - np.trace(R_diff)
                
            if pos_err < 0.02 and rot_err < 0.1:
                rospy.loginfo("✅ Target Reached!")
                self.stop()
                break
                
            # 연산 시간 체크 (디버깅용)
            calc_time = rospy.get_time() - start_t
            if calc_time > 0.025:
                rospy.logwarn(f"Slow loop: {calc_time:.4f}s")
                
            rate.sleep()

if __name__ == "__main__":
    Gen3LiteMPPINodeGPU().main()