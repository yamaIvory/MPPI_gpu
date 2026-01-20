import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time

import torch # 텐서 변환용

# Pinocchio & Meshcat (시각화용)
import pinocchio as pin
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.visualize import MeshcatVisualizer

# [핵심] GPU 솔버 임포트
from mppi_solver_gpu import MPPIControllerGPU

def run_simulation():
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
    mesh_dir = current_dir # meshes 폴더가 있는 위치

    print(f"🚀 [Simulation] GPU Joint-Space MPPI 초기화 중...")
    
    # 2. MPPI 컨트롤러 생성 (GPU)
    # 내부적으로 dynamics_gpu.py를 로딩합니다.
    mppi = MPPIControllerGPU(urdf_path)
    device = mppi.device
    
    # 3. 시각화용 모델 생성 (Pinocchio - CPU)
    # Pinocchio는 화면에 그림 그리는 용도로만 씁니다.
    model = pin.buildModelFromUrdf(urdf_path)
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError:
        print("Error: Meshcat을 열 수 없습니다. 브라우저가 안 열리면 주소를 수동으로 입력하세요.")
    
    viz.loadViewerModel()
    print("🎨 3D 뷰어 로딩 완료! (브라우저 확인)")

    # 4. 초기 상태 설정
    nq = model.nq
    q_curr = np.zeros(nq)
    # 초기 자세 (안전한 홈 포지션)
    q_curr[:6] = np.array([0.0, -0.28, 1.3, 0.0, 0.5, 0.0])

    # 초기 End-Effector 위치 계산 (Target 설정을 위해)
    data = model.createData()
    pin.framesForwardKinematics(model, data, q_curr)
    
    # URDF의 마지막 링크 이름 확인 (보통 DUMMY 또는 END_EFFECTOR)
    # 사용자 URDF에 맞춰 "DUMMY" 사용
    try:
        ee_id = model.getFrameId("DUMMY")
    except:
        ee_id = model.nframes - 1 # 못 찾으면 그냥 마지막 프레임
        
    start_P = data.oMf[ee_id].translation
    start_R = data.oMf[ee_id].rotation
    
    # 목표: 현재 위치에서 Z축으로 +15cm, Y축으로 -10cm
    target_P = start_P.copy()   
    target_P[2] += 0.15 
    target_R = start_R.copy() # 회전은 유지

    # 5. 시각화 객체 (빨간 공 = 목표, 파란 공 = 현재 손)
    viz.viewer['target_ball'].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xff0000, opacity=0.6))
    viz.viewer['target_ball'].set_transform(tf.translation_matrix(target_P))
    viz.viewer['target_frame'].set_object(g.triad(0.15))
    T_target = np.eye(4)
    T_target[:3, :3] = target_R
    T_target[:3, 3] = target_P
    viz.viewer['target_frame'].set_transform(T_target)
    
    viz.viewer['ee_ball'].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0x0000ff, opacity=0.8))
    viz.viewer['ee_frame'].set_object(g.triad(0.15))

    print(f"\n=== 시뮬레이션 시작 ===")
    print(f"   Device: {device}")
    print(f"   Target: {target_P}")
    
    viz.display(q_curr)
    time.sleep(1.0) # 잠시 대기

    try:
        dt = 0.02 # 50Hz
        for step in range(1000):
            loop_start = time.time()
            
            # ---------------------------------------------------------
            # (1) GPU MPPI 계산 (핵심)
            # ---------------------------------------------------------
            # 입력: 현재 각도(numpy), 목표 위치
            # 출력: "최적 관절 속도" (Joint Velocity) -> numpy
            u_opt = mppi.get_optimal_command(q_curr[:6], target_P, target_R)
            
            # ---------------------------------------------------------
            # (2) 로봇 상태 업데이트 (Physics Simulation)
            # ---------------------------------------------------------
            # 우리가 바꾼 DynamicsGPU는 "Joint Velocity"를 받아서 적분합니다.
            # 정확한 시뮬레이션을 위해 GPU Dynamics의 step 함수를 그대로 씁니다.
            
            # Numpy -> Tensor 변환
            q_t = torch.tensor(q_curr[:6], device=device).float().unsqueeze(0)
            u_t = torch.tensor(u_opt, device=device).float().unsqueeze(0)

            with torch.no_grad():
                # step 함수 반환값: q_next, ee_pos, ee_rot (3개)
                q_next_t, _, _ = mppi.dyn.step(q_t, u_t)
            
            # Tensor -> Numpy 변환 (다음 스텝을 위해)
            q_curr[:6] = q_next_t.cpu().numpy().flatten()
            
            # ---------------------------------------------------------
            # (3) 화면 업데이트 (Pinocchio FK)
            # ---------------------------------------------------------
            viz.display(q_curr)
            
            # 현재 EE 위치 확인 (거리 오차 계산용)
            pin.framesForwardKinematics(model, data, q_curr)
            curr_P = data.oMf[ee_id].translation
            curr_R = data.oMf[ee_id].rotation
            
            # 마커 이동
            viz.viewer['ee_ball'].set_transform(tf.translation_matrix(curr_P))
            viz.viewer['ee_frame'].set_transform(data.oMf[ee_id].np) # 현재 프레임 회전 반영

            # 오차 계산
            dist = np.linalg.norm(curr_P - target_P)
            rot_err = 3.0 - np.trace(target_R.T @ curr_R)
            
            # 로그 출력 (10번마다)
            if step % 10 == 0:
                print(f"[Step {step:03d}] P_err: {dist:.4f}m | R_err: {rot_err:.4f}")

            # 종료 조건
            if dist < 0.02:
                print(f"\n✅ 목표 도달 완료! (소요 시간: {step*dt:.4f}s)")
                break

            # ---------------------------------------------------------
            # (4) 리얼타임 싱크 (속도 조절)
            # ---------------------------------------------------------
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n⏹️ 시뮬레이션 종료.")

if __name__ == "__main__":
    run_simulation()