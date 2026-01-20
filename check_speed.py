import torch
import time
import numpy as np
from mppi_solver_gpu import MPPIControllerGPU

def check_speed():
    # 1. 초기화
    print("Initializing MPPI on GPU...")
    try:
        # URDF 경로 확인 필요
        mppi = MPPIControllerGPU("gen3_lite.urdf")
    except Exception as e:
        print(f"초기화 실패: {e}")
        return

    device = mppi.device
    print(f"✅ Device: {device}")
    
    if device == 'cpu':
        print("⚠️ 경고: GPU가 감지되지 않았습니다! CPU로는 0.02초 불가능합니다.")

    # 더미 데이터 생성
    q_curr = np.zeros(6)
    target_P = np.array([0.5, 0.0, 0.5])
    target_R = np.eye(3)

    # 2. 웜업 (Warm-up) - GPU 예열
    # 처음 실행은 메모리 할당 때문에 느리므로 제외
    print("Warm-up (GPU 예열 중)...")
    for _ in range(10):
        mppi.get_optimal_command(q_curr, target_P, target_R)

    # 3. 실제 속도 측정
    iter_count = 100
    print(f"Measuring speed over {iter_count} iterations...")
    
    times = []
    for i in range(iter_count):
        # GPU 시간 측정은 torch.cuda.Event를 써야 정확함
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            mppi.get_optimal_command(q_curr, target_P, target_R)
            
            end_event.record()
            torch.cuda.synchronize() # GPU 연산 끝날 때까지 대기
            elapsed = start_event.elapsed_time(end_event) / 1000.0 # ms -> sec
        else:
            start = time.time()
            mppi.get_optimal_command(q_curr, target_P, target_R)
            elapsed = time.time() - start
            
        times.append(elapsed)

    # 4. 결과 출력
    avg_time = np.mean(times)
    max_time = np.max(times)
    freq = 1.0 / avg_time

    print("\n" + "="*40)
    print(f"   K={mppi.K}, N={mppi.N} 성능 측정 결과")
    print("="*40)
    print(f"평균 연산 시간 : {avg_time:.4f} 초")
    print(f"최대 연산 시간 : {max_time:.4f} 초")
    print(f"가능한 주파수  : {freq:.1f} Hz")
    print("-" * 40)
    
    if avg_time <= 0.02:
        print("🚀 [성공] 0.02초 이내입니다! 50Hz 제어 가능합니다.")
    else:
        print(f"⚠️ [주의] 0.02초를 초과했습니다. (목표: 0.02s, 실제: {avg_time:.4f}s)")
        print("    -> K를 줄이거나, 제어 주기를 낮추세요.")

if __name__ == "__main__":
    check_speed()