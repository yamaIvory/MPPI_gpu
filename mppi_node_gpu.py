#!/usr/bin/env python3
import sys
import os
import rospy
import numpy as np
import torch
from std_msgs.msg import Float64MultiArray
from kortex_driver.srv import *
from kortex_driver.msg import *

# 사용자 정의 GPU 모듈 임포트
try:
    from mppi_solver_gpu import MPPIControllerGPU
except ImportError:
    rospy.logerr("mppi_solver_gpu.py 또는 dynamics_gpu.py를 찾을 수 없습니다.")
    sys.exit()

class Gen3LiteMPPINodeGPU:
    def __init__(self):
        rospy.init_node('gen3_lite_mppi_gpu_node')
        
        # 1. 파라미터 및 경로 설정
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
        
        # 2. GPU 기반 MPPI 컨트롤러 초기화
        self.mppi = MPPIControllerGPU(self.urdf_path)
        self.device = self.mppi.device
        
        self.q_curr_full = None
        self.is_ready = False

        self.setup_kortex_services()
        
        self.sub_feedback = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback", 
            BaseCyclic_Feedback, 
            self.cb_joint_feedback
        )
        
        # [수정] 메시지 타입을 Base_JointSpeeds로 변경하고 기본 토픽으로 발행
        self.pub_vel = rospy.Publisher(
            f"/{self.robot_name}/in/joint_velocity", 
            Base_JointSpeeds, 
            queue_size=1
        )

        rospy.on_shutdown(self.emergency_stop)
        rospy.loginfo(f"✅ MPPI GPU 노드 초기화 완료 (장치: {self.device})")

    def setup_kortex_services(self):
        """Kortex 드라이버 서비스 연결"""
        prefix = f"/{self.robot_name}"
        try:
            rospy.wait_for_service(prefix + '/base/clear_faults', timeout=5.0)
            self.srv_clear_faults = rospy.ServiceProxy(prefix + '/base/clear_faults', Base_ClearFaults)
            self.srv_set_ref_frame = rospy.ServiceProxy(prefix + '/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame)
            self.srv_activate_notif = rospy.ServiceProxy(prefix + '/base/activate_publishing_of_action_topic', OnNotificationActionTopic)
            rospy.loginfo("🔗 Kortex 서비스 연결 성공")
        except rospy.ROSException:
            rospy.logerr("❌ Kortex 서비스를 찾을 수 없습니다. 드라이버가 실행 중인지 확인하세요.")

    def cb_joint_feedback(self, msg):
        """로봇 피드백 콜백 (Degree -> Radian 변환 및 10차원 벡터 구성)"""
        # 팔 관절 6개 수신
        q_arm = [np.deg2rad(msg.actuators[i].position) for i in range(6)]
        
        # 전체 관절 상태 업데이트 (그리퍼 4개 포함 10차원)
        q_full = np.zeros(10)
        q_full[:6] = q_arm
        # 그리퍼 관절(6~9)은 현재 0으로 고정 (필요시 추가 피드백 할당)
        
        self.q_curr_full = q_full
        self.is_ready = True

    def emergency_stop(self):
        """노드 종료 시 로봇 즉시 정지 (Base_JointSpeeds 형식)"""
        rospy.logwarn("⚠️ 시스템 종료: 로봇 정지 명령 발행")
        msg = Base_JointSpeeds()
        for i in range(6):
            js = JointSpeed()
            js.joint_identifier = i
            js.value = 0.0
            js.duration = 0
            msg.joint_speeds.append(js)
        self.pub_vel.publish(msg)

    def hardware_init(self):
        """로봇 하드웨어 초기화 (결함 제거 및 참조 프레임 설정)"""
        rospy.loginfo("🛠️ 하드웨어 초기화 중...")
        self.srv_clear_faults()
        
        ref_req = SetCartesianReferenceFrameRequest()
        ref_req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        self.srv_set_ref_frame(ref_req)
        
        self.srv_activate_notif(OnNotificationActionTopicRequest())
        rospy.sleep(1.0)
        return True

    def control_loop(self, target_P, target_R):
        """실제 MPPI 제어 루프 (Base_JointSpeeds 및 안전장치 적용)"""
        hz = 50 
        rate = rospy.Rate(hz) 
        dt_period = 1.0 / hz
        
        rospy.loginfo(f"🚀 목표 지점으로 이동 시작: {target_P}")

        while not rospy.is_shutdown():
            if not self.is_ready or self.q_curr_full is None:
                continue

            start_time = rospy.get_time()

            # 1. MPPI 최적 속도 계산
            u_opt = self.mppi.get_optimal_command(self.q_curr_full[:6], target_P, target_R)

            # 2. IK 변환
            q_tensor = torch.as_tensor(self.q_curr_full[:6], device=self.device).float().unsqueeze(0)
            u_tensor = torch.as_tensor(u_opt, device=self.device).float().unsqueeze(0)
            
            with torch.no_grad():
                dq_tensor = self.mppi.dyn.solve_ik_batch(q_tensor, u_tensor)
                dq_arm = dq_tensor.squeeze(0).cpu().numpy()

            # [안전장치] 연산 시간 체크
            calc_time = rospy.get_time() - start_time
            if calc_time > dt_period:
                rospy.logwarn_throttle(1, f"⚠️ 연산 지연: {calc_time:.3f}s")
                self.emergency_stop()
                rate.sleep()
                continue

            # 3. 단위 변환: Rad/s -> Deg/s (Kinova Native 방식 필수)
            dq_deg = np.rad2deg(dq_arm)
            dq_deg = np.clip(dq_deg, -20.0, 20.0) # 도/초 단위로 클리핑

            # 4. 명령 발행 (Base_JointSpeeds 형식)
            msg = Base_JointSpeeds()
            for i in range(6):
                js = JointSpeed()
                js.joint_identifier = i
                js.value = dq_deg[i]
                js.duration = 20 # 0.02초 동안 유효
                msg.joint_speeds.append(js)
            
            self.pub_vel.publish(msg)

            # 5. 거리 확인 (생략 가능하나 유지를 위해 남김)
            with torch.no_grad():
                _, curr_P_tensor, _, _ = self.mppi.dyn.step(q_tensor, torch.zeros_like(u_tensor))
                curr_P = curr_P_tensor.squeeze().cpu().numpy()
            
            if np.linalg.norm(curr_P - target_P) < 0.02:
                rospy.loginfo("🎯 목표 도달 완료!")
                self.emergency_stop()
                break

            rate.sleep()

    def main(self):
        if not self.hardware_init():
            return

        # ---------------------------------------------------------
        # [추가] GPU 웜업 (Warm-up): 출발 전에 미리 계산해보기
        # ---------------------------------------------------------
        rospy.loginfo("🔥 GPU 예열 중... (잠시 대기)")
        dummy_q = np.zeros(6)
        dummy_target = np.array([0.5, 0.0, 0.5])
        # 가짜로 10번 정도 계산해서 캐시를 채웁니다.
        for _ in range(10):
            self.mppi.get_optimal_command(dummy_q, dummy_target, np.eye(3))
        rospy.loginfo("✅ GPU 예열 완료! 제어를 시작합니다.")
        # ---------------------------------------------------------
        
        rospy.loginfo("⌛ 로봇 피드백 대기 중...")
        while not self.is_ready and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 현재 위치에서 위로 7cm 상승하는 목표 설정
        q_tensor = torch.as_tensor(self.q_curr_full[:6], device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            _, start_P, start_R, _ = self.mppi.dyn.step(q_tensor, torch.zeros((1, 6), device=self.device))
        
        target_P = start_P.squeeze().cpu().numpy()
        target_P[2] += 0.07  # 7cm 위로
        target_R = start_R.squeeze().cpu().numpy()

        self.control_loop(target_P, target_R)

if __name__ == "__main__":
    try:
        node = Gen3LiteMPPINodeGPU()
        node.main()
    except rospy.ROSInterruptException:
        pass