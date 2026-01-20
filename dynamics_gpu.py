import torch
import pytorch_kinematics as pk
import numpy as np

class DynamicsGPU:
    def __init__(self, urdf_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Dynamics Device: {self.device}")

        # 1. URDF 로드 (pk 사용)
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        
        # [중요] URDF 파일(gen3_lite.urdf)에 명시된 링크 이름 사용
        # root="BASE", end="DUMMY"
        self.chain = pk.build_serial_chain_from_urdf(
            urdf_data, "DUMMY", "BASE"
        )
        self.chain = self.chain.to(device=self.device)
        self.n_dof = len(self.chain.get_joint_parameter_names())

        # 2. 파라미터 (Tensor 변환)
        self.dt = 0.02
        self.damping = 1e-4
        
        # 관절 한계 (Batch 계산을 위해 shape=(1, n_dof)로 만듦)
        # Gen3 Lite는 무한 회전 관절이 많지만, 안전을 위해 -2pi ~ 2pi 설정
        self.q_min = torch.tensor([-6.28] * self.n_dof, device=self.device).float().view(1, -1)
        self.q_max = torch.tensor([ 6.28] * self.n_dof, device=self.device).float().view(1, -1)

    def solve_ik_batch(self, q, u_task):
        """
        [핵심] 500개의 IK를 동시에 풉니다 (Batch DLS)
        q: (Batch, n_dof)
        u_task: (Batch, 6)
        """
        B = q.shape[0]
        
        # 1. Jacobian 계산 (Batch 지원)
        J = self.chain.jacobian(q)  # (B, 6, n_dof)
        
        # 2. DLS IK: dq = J.T * (J*J.T + lambda^2*I)^-1 * u
        # J @ J.T 계산
        JJT = torch.matmul(J, J.transpose(-1, -2)) # (B, 6, 6)
        
        # Damping Identity Matrix 추가
        damp_eye = (self.damping**2) * torch.eye(6, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Linear Solve (Ax = B) -> GPU 병렬 연산
        # u_task를 (B, 6, 1)로 맞춰줘야 함
        u_input = u_task.unsqueeze(-1) 
        temp = torch.linalg.solve(JJT + damp_eye, u_input)
        
        # dq = J.T * temp
        dq = torch.matmul(J.transpose(-1, -2), temp).squeeze(-1) # (B, n_dof)

        # 3. [Safety] --------------속도 제한---------------------------------------
        joint_vel_limit = 0.1
        
        max_vel = torch.max(torch.abs(dq), dim=1, keepdim=True).values # (B, 1)
        scale = torch.clamp(joint_vel_limit / (max_vel + 1e-8), max=1.0)
        dq = dq * scale
        
        return dq

    def step(self, q_curr, u_task):
        """
        q_curr: (Batch, n_dof)
        u_task: (Batch, 6)
        """
        # 1. Batch IK
        dq = self.solve_ik_batch(q_curr, u_task)
        
        # 2. 적분
        q_next = q_curr + dq * self.dt
        
        # 3. 위치 제한
        q_next = torch.max(torch.min(q_next, self.q_max), self.q_min)
        
        # 4. FK (Batch)
        tg = self.chain.forward_kinematics(q_next)
        m = tg.get_matrix() # (B, 4, 4)
        
        pos = m[:, :3, 3] # (B, 3)
        rot = m[:, :3, :3] # (B, 3, 3)
        
        return q_next, pos, rot, dq