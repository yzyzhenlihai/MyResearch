import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.tianshou.tianshou.policy import BasePolicy
from src.tianshou.tianshou.data import Batch, ReplayBuffer, to_torch_as
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import swanlab as wandb
class VectorField(nn.Module):
    """(Teacher) Flow Matching Vector Field: v = f(s, a, t)"""
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        # 输入: State + Action + Time
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_sizes[0]),
            nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.GELU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )

    def forward(self, state, action, t):
        if isinstance(t, float):
            t = torch.full((action.shape[0], 1), t, device=action.device)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        
        inp = torch.cat([state, action, t], dim=-1)
        return self.net(inp)

class OneStepActor(nn.Module):
    """(Student) One-step Generator: a = \pi(s, z)"""
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        # 输入: State + Noise
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.GELU(),
            nn.Linear(hidden_sizes[1], action_dim)
            #nn.Tanh() # 假设 Action Embedding 是归一化的或在 [-1, 1] 之间
        )

    def forward(self, state, noise):
        inp = torch.cat([state, noise], dim=-1)
        return self.net(inp)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),  # <--- 加入 LayerNorm
            nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),  # <--- 加入 LayerNorm
            nn.GELU(),
            nn.Linear(hidden_sizes[1], 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.GELU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, state, action):
        """同时返回 Q1 和 Q2 的值"""
        inp = torch.cat([state, action], dim=-1)
        q1 = self.q1(inp)
        q2 = self.q2(inp)
        return q1, q2
    
    def q1_forward(self, state, action):
        """只计算 Q1 (用于 Actor 更新时，通常只用 Q1 或 min(Q1, Q2))"""
        inp = torch.cat([state, action], dim=-1)
        return self.q1(inp)
    
class FQLPolicy(BasePolicy):
    """
    Flow Q-Learning Policy adapted for Recommender Systems.
    """
    def __init__(
        self,
        actor_flow: nn.Module,      # Teacher
        actor_onestep: nn.Module,   # Student
        critic: nn.Module,          # Critic
        optim: Tuple[torch.optim.Optimizer, torch.optim.Optimizer], # (optim_RL, optim_state)
        state_tracker,
        action_dim: int,
        alpha: float = 10.0,    # 蒸馏权重
        alpha_end: float = 1.0,
        decay_steps: int = 100000, # 持续衰减的步数（通常设为总训练步数）
        flow_steps: int = 10,   # ODE Solver 步数
        discount_factor: float = 0.99,
        tau: float = 0.005,     # 软更新系数
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        device: Union[str, torch.device] = "cpu",
        lambda_div: float = 0.1,

        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor_flow = actor_flow
        self.actor_onestep = actor_onestep
        self.critic = critic
        # 创建 Target Critic
        self.critic_target = type(critic)(
            state_tracker.emb_dim+1, action_dim
        ).to(next(critic.parameters()).device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.optim_RL, self.optim_state = optim
        self.state_tracker = state_tracker
        self.action_dim = action_dim
        self._gamma = discount_factor
        self.alpha = alpha
        self.alpha_end = alpha_end
        self.alpha_start = alpha
        self.decay_steps = decay_steps
        self.global_step_counter = 0 # 全局计数器
        self.flow_steps = flow_steps
        self.tau = tau
        self._rew_norm = reward_normalization
        self.lambda_div = lambda_div
        # 你的代码中 FQL 是连续控制，所以这里强制标记
        self.action_type = "continuous" 

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        # FQL 是 Off-policy 算法，主要依赖 Q-Learning。
        # 这里为了兼容接口，可以保留简单的处理，或者像 DQN 一样计算 n-step return。
        # 简单起见，我们直接返回 batch，具体的 Q 计算在 learn 中进行。
        return batch

    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: np.ndarray = None,
        is_obs = None,
        is_train = True, 
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        use_batch_in_statetracker = False,
        **kwargs: Any,
    ) -> Batch:
        """
        推理阶段：生成推荐动作
        """
        # 1. 使用 State Tracker 获取状态 Embedding
        obs_emb = self.state_tracker(
            buffer=buffer, 
            indices=indices, 
            is_obs=is_obs, 
            batch=batch, 
            is_train=is_train, 
            use_batch_in_statetracker=use_batch_in_statetracker
        )
        
        # 2. 采样噪声 (Latent Exploration)
        # 这里的 noise 类似于 A2C 中的 logits 随机性，但在 FQL 中是输入的 latent 变量
        batch_size = obs_emb.shape[0]
        noise = torch.randn(batch_size, self.action_dim, device=obs_emb.device)
        
        # 3. 使用 Student 网络一步生成动作
        act = self.actor_onestep(obs_emb, noise)
        act = torch.clamp(act, -1, 1)
        # 构造返回结果，RecPolicy 会接手处理 act 到 item_id 的映射
        return Batch(act=act, state=state, logits=act) # logits在这里仅作占位

    def _solve_flow_ode(self, state, batch_size):
        """Teacher 模型的推理 (ODE Solver)"""
        x = torch.randn(batch_size, self.action_dim, device=state.device)
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = i * dt
            v = self.actor_flow(state, x, t)
            x = x + v * dt
        return x.detach()
    
    def _solve_flow_ode_with_start_noise(self, state, x_start):
        """Teacher 模型的推理 (ODE Solver)"""
        x = x_start.clone()
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = i * dt
            v = self.actor_flow(state, x, t)
            x = x + v * dt
        return x.detach()

    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        
        losses, critic_losses, flow_losses, distill_losses, q_losses = [], [], [], [], []
        alpha_values = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.global_step_counter += 1
                progress = min(1.0, self.global_step_counter / self.decay_steps)
                self.alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * progress

                self.optim_RL.zero_grad()
                self.optim_state.zero_grad()

                obs = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
                obs_next = self.state_tracker(self._buffer, minibatch.indices, is_obs=False)
                
                # 获取真实动作 (Item Embedding)
                # 注意：这里假设 minibatch.act 已经是 embedding 或者 RecPolicy 处理过的连续向量
                # 如果 minibatch.act 是 item_id，你需要在这里再转换一次，或者保证传入的是 embedding
                # 通常 RecPolicy 的 buffer 里存的是 item_id，这里可能需要 map 一下
                if minibatch.act.dtype in [torch.int64, torch.int32, torch.long]:
                     # 如果 buffer 里存的是 ID，这里需要 lookup embedding
                     # 假设 state_tracker 有 get_embedding 方法
                     real_actions = self.state_tracker.get_embedding(minibatch.act, "action")
                else:
                     real_actions = to_torch_as(minibatch.act, obs)

                rew = to_torch_as(minibatch.rew, obs).view(-1, 1)
                done = to_torch_as(minibatch.done, obs).view(-1, 1)

                # ==============================
                # Loss Part A: Critic Update (Q-Learning)
                # ==============================
                with torch.no_grad():
                    # 计算 Target Q
                    # 使用 Student 网络采样下一个动作
                    next_noise = torch.randn_like(real_actions)
                    next_action = self.actor_onestep(obs_next, next_noise)
                    #target_q = self.critic_target(obs_next, next_action)
                    target_q1, target_q2 = self.critic_target(obs_next, next_action)
                    target_q = torch.min(target_q1, target_q2) # 取最小值，缓解高估
                    td_target = rew + self._gamma * (1.0 - done) * target_q
                
                current_q1, current_q2 = self.critic(obs, real_actions)
                critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

                # ==============================
                # Loss Part B: Flow Matching (Teacher Training)
                # ==============================
                # 训练 Teacher 拟合真实数据分布 (Behavior Cloning)
                x0 = torch.randn_like(real_actions)
                x1 = real_actions
                t = torch.rand(len(obs), 1, device=obs.device)
                
                xt = (1 - t) * x0 + t * x1
                target_v = x1 - x0
                pred_v = self.actor_flow(obs, xt, t)
                
                flow_loss = F.mse_loss(pred_v, target_v)

                # ==============================
                # Loss Part C: Distillation (Student mimics Teacher)
                # ==============================
                # Student 试图一步预测 Teacher (ODE) 的结果
                z1 = torch.randn_like(real_actions)
                z2 = torch.randn_like(real_actions)
                with torch.no_grad():
                    teacher_target = self._solve_flow_ode_with_start_noise(obs, z1) # 传入 相同的噪声，而不是内部随机生成
                
                # 冻结critic参数，让actor梯度不流到critic
                for p in self.critic.parameters():
                    p.requires_grad = False
                
                student_pred1 = self.actor_onestep(obs, z1)
                student_pred2 = self.actor_onestep(obs, z2)
                diversity_loss = -torch.mean(torch.norm(student_pred1 - student_pred2, dim=-1))
                distill_loss = F.mse_loss(student_pred1, teacher_target) + self.lambda_div * diversity_loss
                

                # ==============================
                # Loss Part D: Q-Maximization (Student maximizes Reward)
                # ==============================
                # 让 Student 生成的动作尽可能分数高
                # 重新生成一个动作用于梯度回传
                
                student_action_clipped = torch.clamp(student_pred1, -1, 1) # 网络没有加tanh，需要手动截断
                q_values = self.critic.q1_forward(obs, student_action_clipped) # 只用q1指导梯度
                raw_q_loss = -q_values.mean()
                # Q loss归一化技巧
                abs_q_mean = torch.abs(q_values).mean()
                lam = 1.0 / (abs_q_mean.detach() + 1e-6)
                q_loss_val = lam * raw_q_loss
                # 解冻critic 参数
                for p in self.critic.parameters():
                    p.requires_grad = True
                # --- Total Loss ---
                total_loss = critic_loss + flow_loss + self.alpha * distill_loss + q_loss_val

                total_loss.backward()
                self.optim_RL.step()
                self.optim_state.step()

                # --- Soft Update ---
                self.sync_weight()

                # Logging
                losses.append(total_loss.item())
                critic_losses.append(critic_loss.item())
                flow_losses.append(flow_loss.item())
                distill_losses.append(distill_loss.item())
                q_losses.append(q_loss_val.item())
                alpha_values.append(self.alpha)

        wandb.log({"FQL/total_loss": np.mean(losses),
                   "FQL/critic_loss": np.mean(critic_losses),
                   "FQL/flow_loss": np.mean(flow_losses),
                   "FQL/distill_loss": np.mean(distill_losses),
                   "FQL/q_loss_val": np.mean(q_losses),
                   "FQL/raw_q_loss": raw_q_loss,
                   "FQL/alpha": np.mean(alpha_values)}) # policy loss 最大化Q值
        return {
            "loss": losses,
            "loss/critic": critic_losses,
            "loss/flow": flow_losses,
            "loss/distill": distill_losses,
            "loss/q": q_losses,
        }

    def sync_weight(self) -> None:
        """Soft update target network"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)