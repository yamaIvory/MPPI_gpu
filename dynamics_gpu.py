import torch
import pytorch_kinematics as pk
import numpy as np

class DynamicsGPU:
    def __init__(self, urdf_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. URDF 로드 (FK 계산용)
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        
        self.chain = pk.build_serial_chain_from_urdf(urdf_data, "DUMMY", "BASE")
        self.chain = self.chain.to(device=self.device)
        self.n_dof = len(self.chain.get_joint_parameter_names())
        
        self.dt = 0.05
        
        # 관절 한계 (Gen3 Lite 하드웨어 리밋)
        # 안전을 위해 약간의 마진을 둡니다.
        self.q_min = torch.tensor([-2.6]*6, device=self.device).float().view(1, -1)
        self.q_max = torch.tensor([ 2.6]*6, device=self.device).float().view(1, -1)

    def step(self, q_curr, u_joint_vel):
        """
        [Joint-Space Dynamics]
        q_curr: (Batch, 6) - 현재 관절 각도
        u_joint_vel: (Batch, 6) - 제어 입력 (관절 속도)
        """
        # 1. Dynamics: 단순 적분 (매우 빠름, 텐서 덧셈)
        q_next = q_curr + u_joint_vel * self.dt
        
        # 2. 관절 위치 제한 (Clamping)
        q_next = torch.max(torch.min(q_next, self.q_max), self.q_min)
        
        # 3. Cost 계산을 위해 FK 수행 (Batch Forward Kinematics)
        # IK와 달리 역행렬 계산이 없어서 GPU에서 순식간에 처리됨
        tg = self.chain.forward_kinematics(q_next)
        m = tg.get_matrix() # (Batch, 4, 4)
        
        ee_pos = m[:, :3, 3] # (Batch, 3)
        ee_rot = m[:, :3, :3] # (Batch, 3, 3)
        
        return q_next, ee_pos, ee_rot