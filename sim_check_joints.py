import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time
import sys

# Pinocchio 관련 임포트
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# ---------------------------------------------------------
# 운영체제 확인 및 키보드 입력 라이브러리 설정
# ---------------------------------------------------------
if os.name == 'nt':  # Windows인 경우
    import msvcrt
else:                # Linux/Mac인 경우
    import termios
    import tty
    import select

# ---------------------------------------------------------
# 키보드 입력 함수 (Cross-Platform)
# ---------------------------------------------------------
def get_key():
    """
    운영체제에 맞춰 키 입력을 받아오는 함수
    """
    if os.name == 'nt':  # Windows
        # 키보드가 눌렸는지 확인 (Non-blocking)
        if msvcrt.kbhit():
            # 눌린 키 읽기 (bytes -> string 변환 필요)
            try:
                key = msvcrt.getch().decode('utf-8').lower()
                return key
            except UnicodeDecodeError:
                return ''
        return ''
    
    else:  # Linux/Mac
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        return key

def run_joint_check():
    # 1. URDF 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
    mesh_dir = current_dir

    print("🚀 [Simulation] URDF 로딩 중...")
    # URDF 파일 존재 여부 확인 (디버깅용)
    if not os.path.exists(urdf_path):
        print(f"Error: URDF 파일을 찾을 수 없습니다: {urdf_path}")
        return

    model = pin.buildModelFromUrdf(urdf_path)
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)

    # 2. 뷰어 실행
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError:
        print("Error: 브라우저를 수동으로 열어주세요.")
    
    viz.loadViewerModel()
    
    # 3. 초기 상태 (홈 포지션)
    # 3. 초기 상태 (홈 포지션)
    # 모델의 전체 관절 수(nq)에 맞춰 0으로 초기화된 배열을 만듭니다.
    q_home = np.zeros(model.nq)
    
    # 앞쪽 6개(로봇 팔)만 원하는 각도로 설정하고, 나머지(그리퍼 등)는 0으로 둡니다.
    # 만약 model.nq가 6보다 작다면 에러가 나지 않도록 안전장치를 둡니다.
    arm_joints = [0.0, -0.5, 1.5, 0.0, 0.0, 0.0]
    q_home[:len(arm_joints)] = arm_joints
    
    # (선택사항) 관절 이름 확인용 출력
    print(f"ℹ️  Model Joint Count: {model.nq}")
    for i, name in enumerate(model.names):
        print(f"  - Joint {i}: {name}")
        
    viz.display(q_home)

    print("\n" + "="*50)
    print("🤖 시뮬레이션(URDF) 관절 방향 확인 모드")
    print("="*50)
    print("숫자 키를 누르면 해당 관절이 시뮬레이션 상의 [+] 방향으로 회전합니다.")
    print("--------------------------------------------------")
    print(" [0] : 1번 관절 (Base)")
    print(" [1] : 2번 관절 (Shoulder) -> ★여기를 잘 보세요")
    print(" [2] : 3번 관절 (Elbow)")
    print(" [3] : 4번 관절 (Wrist 1)")
    print(" [4] : 5번 관절 (Wrist 2)")
    print(" [5] : 6번 관절 (Wrist 3)")
    print("--------------------------------------------------")
    print(" [r] : 초기 위치(Reset)")
    print(" [q] : 종료")
    print("="*50)

    q_curr = q_home.copy()

    # 리눅스용 termios 설정 백업 (윈도우에선 무시)
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    try:
        while True:
            key = get_key()
            
            # 윈도우에서는 루프가 너무 빨리 도는 것을 방지하기 위해 짧은 대기
            if os.name == 'nt' and key == '':
                time.sleep(0.1)

            if key == 'q':
                print("종료합니다.")
                break
            
            elif key == 'r':
                print("🔄 Reset Position")
                q_curr = q_home.copy()
                viz.display(q_curr)

            elif key in ['0', '1', '2', '3', '4', '5']:
                idx = int(key)
                print(f"▶️  Moving Joint [{idx}] in (+) Direction...")
                
                # 애니메이션
                target_val = q_curr[idx] + 0.1 
                start_val = q_curr[idx]
                
                steps = 30
                for i in range(steps):
                    alpha = (i + 1) / steps
                    q_curr[idx] = start_val + (target_val - start_val) * alpha
                    viz.display(q_curr)
                    time.sleep(0.02)
                    
                print(f"   Done. (Current Angle: {q_curr[idx]:.2f} rad)\n")

    except Exception as e:
        print(f"Error 발생: {e}")
    finally:
        # 리눅스 터미널 설정 복구 (윈도우에선 실행 안 함)
        if os.name != 'nt' and settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == "__main__":
    run_joint_check()