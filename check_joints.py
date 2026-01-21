#!/usr/bin/env python3
import rospy
import sys
import select
import termios
import tty
from kortex_driver.msg import Base_JointSpeeds, JointSpeed

def get_key():
    """키보드 입력을 한 글자씩 받아오는 함수 (엔터 없이)"""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def test_joints():
    rospy.init_node('check_all_joints_direction')
    
    # 로봇 이름 (사용자 환경에 맞게 수정)
    robot_name = rospy.get_param('~robot_name', "my_gen3")
    pub = rospy.Publisher(f"/{robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=1)
    
    rospy.sleep(1.0) # 연결 대기

    print("\n" + "="*50)
    print("🤖 Gen3 Lite 관절 방향 테스트 모드")
    print("="*50)
    print("각 숫자 키를 누르면 해당 관절이 '+10도/초'로 1.5초간 움직입니다.")
    print("--------------------------------------------------")
    print(" [0] : 1번 관절 (Base)")
    print(" [1] : 2번 관절 (Shoulder) -> 아까 문제였던 곳")
    print(" [2] : 3번 관절 (Elbow)")
    print(" [3] : 4번 관절 (Wrist 1)")
    print(" [4] : 5번 관절 (Wrist 2)")
    print(" [5] : 6번 관절 (Wrist 3)")
    print("--------------------------------------------------")
    print(" [q] : 종료")
    print("="*50)
    print("⚠️  주의: 로봇 주변을 비워주세요! (비상정지 준비)\n")

    joint_names = ["J0 (Base)", "J1 (Shoulder)", "J2 (Elbow)", 
                   "J3 (Wrist1)", "J4 (Wrist2)", "J5 (Wrist3)"]

    try:
        while not rospy.is_shutdown():
            key = get_key()
            
            if key == 'q':
                print("테스트를 종료합니다.")
                break
            
            # 숫자 0~5 입력 확인
            if key in ['0', '1', '2', '3', '4', '5']:
                idx = int(key)
                print(f"▶️  Testing [{idx}] {joint_names[idx]} ... (+방향 이동)")
                
                # 명령 생성 (+10 deg/s)
                msg = Base_JointSpeeds()
                js = JointSpeed()
                js.joint_identifier = idx
                js.value = 5.0  # 양수(+) 방향 명령
                js.duration = 0
                msg.joint_speeds.append(js)
                
                # 1초 동안 전송 (안전하게 짧게)
                end_time = rospy.get_time() + 1.0
                rate = rospy.Rate(50) # 50Hz
                
                while rospy.get_time() < end_time:
                    pub.publish(msg)
                    rate.sleep()
                
                # 정지
                stop_msg = Base_JointSpeeds()
                js.value = 0.0
                stop_msg.joint_speeds.append(js)
                pub.publish(stop_msg)
                print("   [정지] 확인 완료. 다음 키를 누르세요.\n")
                
    except Exception as e:
        print(e)
    finally:
        # 안전하게 정지 메시지 보내고 종료
        stop_msg = Base_JointSpeeds()
        for i in range(6):
            js = JointSpeed()
            js.joint_identifier = i
            js.value = 0.0
            stop_msg.joint_speeds.append(js)
        pub.publish(stop_msg)

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    test_joints()
