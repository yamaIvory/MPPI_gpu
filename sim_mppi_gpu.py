import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pinocchio as pin
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.visualize import MeshcatVisualizer

# [변경] GPU 솔버 임포트
from mppi_solver_gpu import MPPIControllerGPU

def run_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
    mesh_dir = current_dir 

    print(f"🚀 GPU 시뮬레이션 초기화 중...")
    
    # 1. MPPI 컨트롤러 생성 (GPU)
    mppi = MPPIControllerGPU(urdf_path)
    
    # 2. 시각화용 모델 생성 (Pinocchio - CPU)
    # [주의] Pinocchio는 로봇 모델 로딩용으로만 씁니다.
    model = pin.buildModelFromUrdf(urdf_path)
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError:
        print("Error: Meshcat을 열 수 없습니다.")
        return
    viz.loadViewerModel()
    print("-> 3D 뷰어 로딩 완료!")

    # 3. 목표물 설정
    nq = model.nq
    q_curr = np.zeros(nq)
    q_curr[:6] = np.array([0.0, -0.28, 1.3, 0.0, 0.5, 3.14])

    # 초기 위치 계산 (Pinocchio 사용)
    data = model.createData()
    pin.framesForwardKinematics(model, data, q_curr)
    ee_id = model.getFrameId("DUMMY") # URDF에 DUMMY가 있으므로 사용
    start_P = data.oMf[ee_id].translation
    start_R = data.oMf[ee_id].rotation
    
    target_P = start_P.copy()   
    target_P[2] += 0.1
    target_R = start_R.copy()

    # 시각화 객체 생성
    viz.viewer['target_ball'].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
    viz.viewer['target_ball'].set_transform(tf.translation_matrix(target_P))
    viz.viewer['ee_ball'].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x0000ff, opacity=0.8))
    viz.viewer['target_frame'].set_object(g.triad(0.1))
    viz.viewer['ee_frame'].set_object(g.triad(0.1))

    print(f"\n=== 시뮬레이션 시작 (K={mppi.K}, Device={mppi.device}) ===")
    
    viz.display(q_curr)
    time.sleep(1.0)

    try:
        dt = 0.02
        for step in range(1000):
            loop_start = time.time()
            
            # (1) GPU MPPI 계산
            # q_curr는 numpy지만 내부에서 자동으로 GPU 텐서로 변환됨
            u_opt = mppi.get_optimal_command(q_curr[:6], target_P, target_R)
            
            # (2) 로봇 이동 (단순 적분 for visualization)
            # 여기서는 시뮬레이션 환경이므로 간단히 오일러 적분 사용
            # 실제로는 mppi.dyn.step을 써도 되지만, 반환값이 텐서라 변환 필요
            
            # IK 계산을 위해 임시로 DynamicsGPU step 함수 활용 (Batch=1)
            import torch
            q_curr_tensor = torch.tensor(q_curr[:6], device=mppi.device).float().unsqueeze(0)
            u_opt_tensor = torch.tensor(u_opt, device=mppi.device).float().unsqueeze(0)

            # 2. 다음 상태 계산 (결과는 6개)
            q_next_tensor, _, _, _ = mppi.dyn.step(q_curr_tensor, u_opt_tensor)

            # 3. 전체 상태 벡터(10개) 중 팔 부분(앞 6개)만 업데이트
            q_curr[:6] = q_next_tensor.cpu().numpy().flatten()
            
            # (3) 화면 업데이트 (Pinocchio FK 사용)
            viz.display(q_curr)
            
            pin.framesForwardKinematics(model, data, q_curr)
            curr_P = data.oMf[ee_id].translation
            curr_R = data.oMf[ee_id].rotation
            
            viz.viewer['ee_ball'].set_transform(tf.translation_matrix(curr_P))
            T_target = np.eye(4)
            T_target[:3, 3] = target_P
            T_target[:3, :3] = target_R
            viz.viewer['target_frame'].set_transform(T_target)
            viz.viewer['ee_frame'].set_transform(data.oMf[ee_id].np)

            # 결과 확인
            dist = np.linalg.norm(curr_P - target_P)
            rot_err = 3.0 - np.trace(target_R.T @ curr_R)
            
            if step % 10 == 0:
                print(f"[Step {step}] 거리: {dist:.4f}m, 회전오차: {rot_err:.4f}")

            if dist < 0.02 and rot_err < 0.1:
                print(f"\n✅ 목표 도달 완료!")
                break

            # 속도 조절
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n종료합니다.")

if __name__ == "__main__":
    run_simulation()