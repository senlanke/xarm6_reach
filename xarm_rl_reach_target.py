import numpy as np
import mujoco
import gymnasium as  gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class PandaObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False):
        super(PandaObstacleEnv, self).__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.model = mujoco.MjModel.from_xml_path('/home/lj402/.mujoco/models/mujoco_menagerie/ufactory_xarm6/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = -120.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.home_joint_pos = np.deg2rad(np.array([0, -90, -30, 0, 30, 0], dtype=np.float32))
        
        self.goal_size = 0.03
        
        # 约束工作空间
        self.workspace = {
            'x': [0.2, 0.6],
            'y': [-0.4, 0.4],
            'z': [0.2, 0.6]
        }

        # 末端姿态
        self.tf = np.array([
                [0, 0, 1],
                [0, -1, 0],
                [1, 0, 0]
            ], dtype=np.float64)
        
        # 动作空间与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 6轴关节角度、目标位置、末端四元数、末端速度
        # 观测空间维度：6 + 6 + 3 + 4 + 6 = 25
        self.obs_size = 25
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )


        
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.goal_threshold = 0.005
        self.current_orient_power = 2.0
        self.prev_dist_to_goal = np.inf
        # 添加阶段性奖励跟踪集合
        self._phase_rewards_given = set()
        self.step_count = 0
        self.max_steps = 1500

    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if 0.3 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5 :
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """渲染目标点"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()
        
        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()        
        
        obs = self._get_observation()
        self.start_t = time.time()

        self.prev_dist_to_goal = np.inf
        self._phase_rewards_given = set()
        self.step_count = 0
        # 重置新添加的属性
        self.min_distance = None
        self.previous_distance = None
        self.previous_joint_velocities = np.zeros(6, dtype=np.float32) 
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        # 1. 关节位置观测（带均匀噪声）
        joint_pos = self.data.qpos[:6].copy().astype(np.float32)
        joint_pos += self.np_random.uniform(-0.01, 0.01, size=joint_pos.shape).astype(np.float32)

        # 2. 关节速度观测（带均匀噪声）
        joint_vel = self.data.qvel[:6].copy().astype(np.float32)
        joint_vel += self.np_random.uniform(-0.01, 0.01, size=joint_vel.shape).astype(np.float32)

        # 3. 姿态命令观测（这里等价于目标末端位姿 ee_pose）
        #   包含目标位置 + 期望姿态四元数
        goal_pos = self.goal.copy().astype(np.float32)
        goal_quat = R.from_matrix(self.tf).as_quat().astype(np.float32)  # [x,y,z,w]

        # 4. 上一动作观测
        last_action = self.prev_action.copy().astype(np.float32)

        # 拼接观测向量
        obs = np.concatenate([
            joint_pos,      # 6
            joint_vel,      # 6
            goal_pos,       # 3
            goal_quat,      # 4
            last_action     # 6
        ])

        return obs


    def quat_rotate_vector(self, quat, vec):
        """
        Rotate a vector by a quaternion using MuJoCo convention.
        
        Args:
            quat: quaternion in MuJoCo format [w, x, y, z]
            vec: 3D vector [x, y, z]
        
        Returns:
            Rotated 3D vector
        """
        w, x, y, z = quat
        vx, vy, vz = vec
        
        # Quaternion rotation: v' = q * v * q^-1
        # Expanded form (Hamilton product)
        t = 2 * np.cross([x, y, z], vec)
        rotated = vec + w * t + np.cross([x, y, z], t)
        
        return rotated

    def convert_mujoco_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """
        将 MuJoCo 四元数 [w, x, y, z] 转换为 SciPy 格式 [x, y, z, w]

        Args:
            quat_wxyz (np.ndarray): MuJoCo 四元数，顺序为 [w, x, y, z]

        Returns:
            np.ndarray: SciPy 四元数，顺序为 [x, y, z, w]
        """
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)



    def _calc_reward(
        self,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
        ee_cvel: np.ndarray,
        ee_quat: np.ndarray,
        joint_angles: np.ndarray,
        action: np.ndarray,
        step_count: int
    ) -> tuple[np.ndarray, float, float]:
        """
        奖励函数：融合 Isaac Lab 风格的五个奖励/惩罚项
        - 末端位置误差惩罚
        - 末端位置误差 tanh 奖励
        - 末端方向误差惩罚（基于四元数）
        - 动作变化率惩罚
        - 关节速度惩罚
        """

        # 计算末端到目标的距离
        dist_to_goal = np.linalg.norm(ee_pos - obj_pos)
        done = False
        truncated = False
        # --- 1. 位置误差惩罚 ---
        pos_penalty = -0.2 * dist_to_goal

        # --- 2. 位置误差 tanh 奖励 ---
        pos_reward_tanh = 0.1 * (1 - np.tanh(dist_to_goal / 0.1))

        # --- 3. 姿态误差惩罚（四元数） ---
        # 将期望姿态矩阵转为四元数
        des_quat = R.from_matrix(self.tf).as_quat()   # [x, y, z, w]
        ee_quat_scipy = self.convert_mujoco_quat(ee_quat)
        curr_rot = R.from_quat(ee_quat_scipy)
        des_rot = R.from_quat(des_quat)
        quat_err = des_rot.inv() * curr_rot
        angle_err = np.linalg.norm(quat_err.as_rotvec())
        orient_penalty = -0.1 * angle_err

        # --- 4. 动作变化率惩罚 ---
        action_rate_penalty = -0.0001 * np.linalg.norm(action - self.prev_action)

        # --- 5. 关节速度惩罚 ---
        joint_vel_penalty = -0.0001 * np.linalg.norm(self.data.qvel[:6])

        # --- 总奖励 ---
        total_reward = pos_penalty + pos_reward_tanh + orient_penalty + action_rate_penalty + joint_vel_penalty

        # --- 成功奖励 ---
        if dist_to_goal <= self.goal_threshold:
            total_reward += 50.0 + 0.01 * (self.max_steps - step_count)
            done = True
        if step_count >= self.max_steps:
            total_reward -= 10.0
            truncated = True

        # 更新状态
        self.prev_action = action.copy()

        return np.array(total_reward, dtype=np.float32), done, dist_to_goal, truncated



 

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        self.step_count += 1
        # 动作缩放
        joint_ranges = self.model.jnt_range[:6]
        scaled_action = np.zeros(6, dtype=np.float32)
        for i in range(6):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        # 每步最大关节增量（弧度）
        # max_step = np.deg2rad(5.0)  # 每步最多 2°
        # delta_q = np.clip(action, -1.0, 1.0).astype(np.float32) * max_step
        # target_q = np.clip(
        #     self.data.qpos[:6] + delta_q,
        #     self.model.jnt_range[:6, 0],
        #     self.model.jnt_range[:6, 1]
        # )       
        
        # 执行动作
        self.data.ctrl[:6] = scaled_action
        # self.data.ctrl[:6] = target_q
        mujoco.mj_step(self.model, self.data)
        
        # 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        ee_cvel = self.data.cvel[self.end_effector_id, :3].copy()
        # rot = R.from_quat(ee_quat)
        # ee_quat_euler_rad = rot.as_euler('xyz')
        reward, terminated, dist_to_goal, truncated = self._calc_reward(ee_pos, self.goal, ee_cvel, ee_quat, self.data.qpos[:6], action, self.step_count)
        collision = False

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 400_000,  # 本次训练的新增步数
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    if resume_from is not None:
        print(f"从模型 {resume_from} 恢复训练")
        model = PPO.load(resume_from, env=env)  # 加载时需传入当前环境
    else:
        POLICY_KWARGS = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,          
            batch_size=2048,       
            n_epochs=5,           
            gamma=0.99,
            learning_rate=1e-3,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_reach_target/" # tensorboard --logdir=./tensorboard/panda_reach_target/ --port=6007
        )
    
    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)
    
    record_gif = False
    frames = [] if record_gif else None
    render_scene = None  
    render_context = None 
    pixel_buffer = None 
    viewport = None
    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    delete_flag_file()
    # TRAIN_MODE = False  # 设为True开启训练模式
    TRAIN_MODE = True
    # MODEL_PATH = "assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v1"
    MODEL_PATH = "assets/model/rl_reach_target_checkpoint/xarm_new_ppo_reach_target_v5"
    RESUME_MODEL_PATH = "assets/model/rl_reach_target_checkpoint/xarm_new_ppo_reach_target_v4"
    # RESUME_MODEL_PATH = None
    if TRAIN_MODE:
        train_ppo(
            n_envs=256,                
            total_timesteps=20_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            resume_from=RESUME_MODEL_PATH
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,
        )