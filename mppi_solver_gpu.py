import torch
import numpy as np
from dynamics_gpu import DynamicsGPU

class MPPIControllerGPU:
    def __init__(self, urdf_path):
        # GPU 사용 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dyn = DynamicsGPU(urdf_path, device=self.device)
        self.nq = self.dyn.n_dof

        # ---- Hyperparameters ----
        self.K = 500            # 샘플 수
        self.N = 30             # 예측 단계
        self.dt = 0.02
        self.lambda_ = 0.6
        self.alpha = 0.3

        # Cost Weights (GPU Tensor)
        self.w_pos = 150.0
        self.w_rot = 20.0
        self.w_vel = 0.01
        self.w_pos_term = 300.0
        self.w_rot_term = 50.0

        # Noise
        self.sigma = torch.tensor([1.0]*3 + [0.5]*3, device=self.device)
        
        # Nominal Control (N, 6)
        self.U = torch.zeros((self.N, 6), device=self.device)
        
        self.desk_height = 0.0

    def compute_cost_batch(self, ee_pos, ee_rot, P_goal, R_goal, u, is_terminal=False):
        """
        Batch Cost Calculation
        """
        # 1. 위치 오차
        pos_err = torch.norm(ee_pos - P_goal, dim=1)**2

        # 2. 회전 오차 (Trace)
        R_diff = torch.matmul(R_goal.transpose(-1, -2), ee_rot)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        rot_err = 3.0 - trace

        # 3. 비용 합산
        w_p = self.w_pos_term if is_terminal else self.w_pos
        w_r = self.w_rot_term if is_terminal else self.w_rot
        
        cost = w_p * pos_err + w_r * rot_err
        
        if not is_terminal:
            ctrl_cost = self.w_vel * torch.sum(u**2, dim=1)
            cost += ctrl_cost

            # 4. [충돌 체크] 높이 제한 (Vectorized)
            collision_mask = ee_pos[:, 2] < self.desk_height
            cost = torch.where(collision_mask, cost + 1e9, cost)

        return cost

    def get_optimal_command(self, q_curr_np, P_goal_np, R_goal_np=None):
        if R_goal_np is None: R_goal_np = np.eye(3)
        
        # Numpy -> Tensor 변환
        q_curr = torch.tensor(q_curr_np, device=self.device).float()
        q_sim = q_curr.unsqueeze(0).repeat(self.K, 1) # (K, n_dof)
        
        P_goal = torch.tensor(P_goal_np, device=self.device).float().unsqueeze(0)
        R_goal = torch.tensor(R_goal_np, device=self.device).float().unsqueeze(0)

        # 노이즈 생성
        noise = torch.randn((self.K, self.N, 6), device=self.device) * self.sigma
        u_samples = self.U.unsqueeze(0) + noise 
        
        # Input Clipping
        u_samples[:, :, :3] = torch.clamp(u_samples[:, :, :3], -0.5, 0.5)
        u_samples[:, :, 3:] = torch.clamp(u_samples[:, :, 3:], -2.0, 2.0)

        total_costs = torch.zeros(self.K, device=self.device)

        # Rollout Loop
        for t in range(self.N):
            u_t = u_samples[:, t, :] # (K, 6)
            q_next, ee_pos, ee_rot, _ = self.dyn.step(q_sim, u_t)
            
            step_c = self.compute_cost_batch(ee_pos, ee_rot, P_goal, R_goal, u_t, is_terminal=False)
            total_costs += step_c * self.dt
            q_sim = q_next

        # Terminal Cost
        term_c = self.compute_cost_batch(ee_pos, ee_rot, P_goal, R_goal, None, is_terminal=True)
        total_costs += term_c

        # Weight Calculation
        beta = torch.min(total_costs)
        weights = torch.exp(-1.0/self.lambda_ * (total_costs - beta))
        weights = weights / (torch.sum(weights) + 1e-10)

        # Update Control Sequence
        delta_u = torch.sum(weights.view(self.K, 1, 1) * noise, dim=0)
        U_new = self.U + delta_u
        self.U = (1 - self.alpha) * self.U + self.alpha * U_new
        
        # Shift & Return
        u_opt_tensor = self.U[0].clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0

        return u_opt_tensor.cpu().numpy()