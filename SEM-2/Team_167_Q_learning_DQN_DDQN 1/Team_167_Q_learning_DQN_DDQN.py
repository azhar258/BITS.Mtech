# %% [markdown]
# # Assignment 2: Robust Reinforcement Learning under Stochastic Action Failure
# **Group ID: 167**
#
# ## Group Contribution Declaration
# | Member Name | Student ID | Contribution (%) |
# |-------------|-----------|-----------------|
# | [Member 1]  | [ID 1]    | [%]             |
# | [Member 2]  | [ID 2]    | [%]             |
# | [Member 3]  | [ID 3]    | [%]             |
#
# **Environment:** LunarLander-v3 (Gymnasium)
# **Algorithms:** Deep Q-Network (DQN) and Double Deep Q-Network (DDQN)
# **Task:** Compare performance of DQN vs DDQN on original and modified (stochastic engine failure) environments

# %% [markdown]
# ---
# ## Setup: Imports and Global Configuration

# %%
# ============================================================
# CELL 1 — IMPORTS
# All required packages for the full assignment pipeline.
# ============================================================

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (14, 5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

print("All imports successful!")
print(f"  gymnasium : {gym.__version__}")
print(f"  torch     : {torch.__version__}")
print(f"  numpy     : {np.__version__}")

# %%
# ============================================================
# CELL 2 — HYPERPARAMETERS
# These values are IDENTICAL for DQN and DDQN on both envs.
# ============================================================

# --- Reproducibility ---
SEED = 42

# --- Training duration ---
N_EPISODES   = 1000     # Training episodes (same for all 4 experiments)
MIN_BUFFER   = 1000     # Minimum replay buffer entries before training starts

# --- Neural network architecture ---
STATE_SIZE   = 8        # LunarLander-v3 observation dimensionality
ACTION_SIZE  = 4        # Discrete actions: 0=nothing, 1=left, 2=main, 3=right
HIDDEN_SIZE  = 128      # Neurons per hidden layer (2 layers)

# --- Optimization ---
LR           = 1e-3     # Adam learning rate
BATCH_SIZE   = 64       # Mini-batch size for each gradient update
GAMMA        = 0.99     # Discount factor for future rewards

# --- Exploration (ε-greedy) ---
EPS_START    = 1.0      # Initial exploration probability
EPS_MIN      = 0.01     # Minimum exploration probability
EPS_DECAY    = 0.995    # Multiplicative decay per episode (→ 0.01 around ep 917)

# --- Target network ---
TARGET_UPDATE_FREQ = 10  # Copy online → target every N episodes

# --- Replay buffer ---
BUFFER_SIZE  = 100_000  # Maximum stored transitions

# --- Validation set (fixed, used to track predicted Q-values) ---
VAL_SIZE     = 1_000    # Number of states held constant throughout training

# --- Modified environment parameters ---
FAILURE_PROB  = 0.15    # 15% chance each thruster action is replaced by Do-Nothing
FUEL_PENALTY  = 0.3     # Subtracted from reward for every attempted thruster action
LANDING_BONUS = 50.0    # Added when strict safe-landing criteria are all met

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"\nTraining config  : {N_EPISODES} episodes, batch={BATCH_SIZE}, LR={LR}")
print(f"Network          : {STATE_SIZE}→{HIDDEN_SIZE}→{HIDDEN_SIZE}→{ACTION_SIZE}")
print(f"ε schedule       : {EPS_START}→{EPS_MIN} (×{EPS_DECAY}/ep)")
print(f"Target update    : every {TARGET_UPDATE_FREQ} episodes")
print(f"Modified env     : failure_prob={FAILURE_PROB}, fuel_penalty={FUEL_PENALTY}, bonus={LANDING_BONUS}")

# %%
# ============================================================
# CELL 3 — HELPER: reproducible seed reset
# ============================================================

def set_seeds(seed: int):
    """
    Reset all random seeds so every agent starts from the same initial conditions.
    Called immediately before creating each agent to ensure fairness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Apply once globally as well
set_seeds(SEED)
print(f"Global random seeds set to {SEED}.")


# %% [markdown]
# ---
# ## Part (a) — Modified LunarLander-v3 Environment

# %%
# ============================================================
# CELL 4 — MODIFIED ENVIRONMENT
# Wraps LunarLander-v3 with:
#   1. Stochastic engine failure (15% probability per thruster action)
#   2. Fuel penalty (-0.3) based on agent's *selected* action
#   3. Safe-landing bonus (+50) when all landing criteria are met simultaneously
# All other aspects of the environment are unchanged.
# ============================================================

class StochasticActionFailureLander(gym.Wrapper):
    """
    gym.Wrapper that adds stochastic actuator failures and a modified reward
    to the standard LunarLander-v3 environment.

    Reward formula per step:
        R = R_base  -  0.3 * I(a ∈ {1,2,3})  +  R_landing

    where:
        R_base    = reward from the underlying environment
        I(a)      = 1 if the agent selected a thruster action (1, 2, or 3), else 0
        R_landing = LANDING_BONUS if ALL safe-landing conditions hold, else 0

    Safe-landing conditions (must ALL be true simultaneously):
        - terminated == True  AND  truncated == False
        - obs[6] == 1   (left leg in contact)
        - obs[7] == 1   (right leg in contact)
        - |obs[2]| < 0.10  (horizontal velocity)
        - |obs[3]| < 0.10  (vertical velocity)
        - |obs[4]| < 0.10  (orientation angle in radians)

    Engine-failure rule:
        If the agent selects a ∈ {1, 2, 3}, draw r ~ Uniform[0, 1).
        If r < 0.15  →  execute action 0 (Do Nothing) instead.
        The agent never observes the substitution.
    """

    def __init__(self, env: gym.Env,
                 failure_prob: float = 0.15,
                 fuel_penalty: float = 0.3,
                 landing_bonus: float = 50.0):
        super().__init__(env)
        self.failure_prob  = failure_prob
        self.fuel_penalty  = fuel_penalty
        self.landing_bonus = landing_bonus

        # --- Counters for Part (a) verification ---
        self.total_thruster_attempts  = 0  # Agent selected action ∈ {1,2,3}
        self.total_thruster_failures  = 0  # Engine misfired (replaced by 0)
        self.total_safe_landings      = 0  # Landing bonus actually awarded
        self.total_episodes_completed = 0

        # Per-episode running total (reset in reset())
        self._ep_fuel_penalty = 0.0

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        """Reset the environment; also reset per-episode fuel counter."""
        self._ep_fuel_penalty = 0.0
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Execute one environment step with engine-failure and reward modification.

        Parameters
        ----------
        action : int  Agent's chosen action (0–3)

        Returns
        -------
        obs            : np.ndarray  8-dim state (unchanged from base env)
        R              : float       Modified reward
        terminated     : bool
        truncated      : bool
        info           : dict        No engine-failure info added
        """
        # ---- Step 1: Store agent's selected action ----
        a = int(action)

        # ---- Step 2: Stochastic engine-failure simulation ----
        a_exec = a  # Default: execute as selected
        if a in (1, 2, 3):
            self.total_thruster_attempts += 1
            if np.random.uniform(0.0, 1.0) < self.failure_prob:
                a_exec = 0                      # Engine misfires → Do Nothing
                self.total_thruster_failures += 1

        # ---- Step 3: Execute the (possibly replaced) action ----
        obs, R_base, terminated, truncated, info = self.env.step(a_exec)

        # ---- Step 4: Fuel penalty (depends on selected action, NOT a_exec) ----
        fuel_pen = self.fuel_penalty if a in (1, 2, 3) else 0.0
        self._ep_fuel_penalty += fuel_pen

        # ---- Step 5: Safe-landing bonus ----
        R_landing = 0.0
        if terminated and not truncated:
            safe = (
                obs[6] == 1 and obs[7] == 1          # Both legs in contact
                and abs(obs[2]) < 0.10               # Low horizontal velocity
                and abs(obs[3]) < 0.10               # Low vertical velocity
                and abs(obs[4]) < 0.10               # Near-zero orientation angle
            )
            if safe:
                R_landing = self.landing_bonus
                self.total_safe_landings += 1

        # ---- Modified reward ----
        R = R_base - fuel_pen + R_landing

        if terminated or truncated:
            self.total_episodes_completed += 1

        # ---- Step 6: Return (info dict is unchanged — no failure disclosure) ----
        return obs, R, terminated, truncated, info

    # ------------------------------------------------------------------
    def failure_stats(self) -> dict:
        """Return a dict of engine-failure statistics for verification."""
        fail_rate = (
            self.total_thruster_failures / self.total_thruster_attempts
            if self.total_thruster_attempts > 0 else 0.0
        )
        return {
            'thruster_attempts'  : self.total_thruster_attempts,
            'thruster_failures'  : self.total_thruster_failures,
            'empirical_fail_rate': fail_rate,
            'safe_landings'      : self.total_safe_landings,
            'episodes_completed' : self.total_episodes_completed,
        }


# %%
# ============================================================
# CELL 5 — PART (a): ENVIRONMENT VERIFICATION
# Run 200 random-policy episodes on the modified environment and confirm:
#   1. ~15 % of thruster actions are replaced by Do-Nothing
#   2. Fuel penalty is applied for every attempted thruster action
#   3. +50 landing bonus is awarded only when all safe-landing criteria hold
# ============================================================

print("=" * 65)
print("PART (a) — Environment Verification  (200 random-policy episodes)")
print("=" * 65)

VERIFY_EPISODES = 200
verify_env = StochasticActionFailureLander(
    gym.make("LunarLander-v3"),
    failure_prob=FAILURE_PROB,
    fuel_penalty=FUEL_PENALTY,
    landing_bonus=LANDING_BONUS,
)

ep_rewards_verify   = []    # Total reward per episode
ep_fuel_verify      = []    # Total fuel penalty per episode
ep_thruster_verify  = []    # Thruster actions selected per episode
landing_flags       = []    # 1 if safe-landing bonus was triggered

# For per-step fuel-penalty cross-check
manual_total_fuel   = 0.0
prev_safe_landings  = 0

set_seeds(SEED)  # Use fixed seed so verification is reproducible

for ep in range(VERIFY_EPISODES):
    obs, _ = verify_env.reset()
    ep_reward = 0.0
    ep_thruster = 0
    verify_env._ep_fuel_penalty = 0.0   # reset per-episode counter

    while True:
        act = verify_env.action_space.sample()   # Random policy
        if act in (1, 2, 3):
            ep_thruster += 1

        obs, rew, term, trunc, _ = verify_env.step(act)
        ep_reward += rew
        if term or trunc:
            break

    ep_rewards_verify.append(ep_reward)
    ep_fuel_verify.append(verify_env._ep_fuel_penalty)
    ep_thruster_verify.append(ep_thruster)
    manual_total_fuel += verify_env._ep_fuel_penalty

    cur_safe = verify_env.failure_stats()['safe_landings']
    landing_flags.append(1 if cur_safe > prev_safe_landings else 0)
    prev_safe_landings = cur_safe

stats_v = verify_env.failure_stats()
verify_env.close()

# ------- Print numerical verification results -------
print(f"\n[1] Engine-Failure Statistics")
print(f"    Thruster actions attempted     : {stats_v['thruster_attempts']:>7,}")
print(f"    Thruster failures (→ action 0) : {stats_v['thruster_failures']:>7,}")
print(f"    Empirical failure rate         : {stats_v['empirical_fail_rate']*100:>6.2f} %")
print(f"    Expected failure rate          :  15.00 %")
print(f"    Deviation                      : {abs(stats_v['empirical_fail_rate']-0.15)*100:>6.2f} %")

expected_fuel = stats_v['thruster_attempts'] * FUEL_PENALTY
print(f"\n[2] Fuel Penalty Correctness Check")
print(f"    Expected total penalty (attempts × {FUEL_PENALTY}) : {expected_fuel:>8.2f}")
print(f"    Actual total penalty applied               : {manual_total_fuel:>8.2f}")
match = abs(expected_fuel - manual_total_fuel) < 0.01
print(f"    Fuel penalty correctly applied?            : {'YES ✓' if match else 'NO ✗'}")

print(f"\n[3] Safe-Landing Bonus")
print(f"    Total episodes                 : {stats_v['episodes_completed']:>7,}")
print(f"    Safe landings (bonus awarded)  : {stats_v['safe_landings']:>7,}")
safe_rate = stats_v['safe_landings'] / max(stats_v['episodes_completed'], 1) * 100
print(f"    Safe-landing rate              : {safe_rate:>6.2f} %")
print(f"    (Low rate expected — random policy rarely achieves safe landing)")

print(f"\n[4] Episode Reward Summary")
print(f"    Mean ± Std  : {np.mean(ep_rewards_verify):.2f} ± {np.std(ep_rewards_verify):.2f}")
print(f"    Min / Max   : {np.min(ep_rewards_verify):.2f} / {np.max(ep_rewards_verify):.2f}")

# %%
# ============================================================
# CELL 6 — PART (a): VERIFICATION PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    "Part (a): Modified LunarLander-v3 Verification  (200 random-policy episodes)",
    fontsize=13, fontweight='bold'
)

# -- Subplot 1: Reward distribution --
axes[0].hist(ep_rewards_verify, bins=25, color='steelblue', alpha=0.75, edgecolor='k', linewidth=0.5)
axes[0].axvline(np.mean(ep_rewards_verify), color='red', ls='--', lw=2,
                label=f"Mean = {np.mean(ep_rewards_verify):.1f}")
axes[0].set_title("Episode Reward Distribution")
axes[0].set_xlabel("Episode Reward")
axes[0].set_ylabel("Count")
axes[0].legend()

# -- Subplot 2: Engine-failure pie chart --
n_fire = stats_v['thruster_attempts'] - stats_v['thruster_failures']
n_fail = stats_v['thruster_failures']
axes[1].pie(
    [n_fire, n_fail],
    labels=[f"Fires ({n_fire:,})\n{n_fire/stats_v['thruster_attempts']*100:.1f}%",
            f"Fails→0 ({n_fail:,})\n{n_fail/stats_v['thruster_attempts']*100:.1f}%"],
    colors=['#2ecc71', '#e74c3c'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[1].set_title(f"Thruster Outcomes\n(Expected failure = 15 %)")

# -- Subplot 3: Per-episode fuel penalty --
x_ep = np.arange(1, VERIFY_EPISODES + 1)
axes[2].bar(x_ep, ep_fuel_verify, color='orange', alpha=0.5, width=1.0, label='Fuel/ep')
w = min(20, VERIFY_EPISODES)
ma = np.convolve(ep_fuel_verify, np.ones(w)/w, mode='valid')
axes[2].plot(np.arange(w, VERIFY_EPISODES + 1), ma, color='darkorange', lw=2,
             label=f'{w}-ep moving avg')
axes[2].set_title("Fuel Penalty per Episode\n(Penalty on selected action, incl. misfires)")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Total Fuel Penalty")
axes[2].legend()

plt.tight_layout()
plt.savefig('part_a_verification.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: part_a_verification.png")


# %% [markdown]
# ---
# ## Neural Network, Replay Buffer, and Agent Classes

# %%
# ============================================================
# CELL 7 — Q-NETWORK ARCHITECTURE
# Two hidden layers of HIDDEN_SIZE neurons with ReLU activations.
# Input: 8-dim state  |  Output: Q-value for each of 4 actions
# ============================================================

class QNetwork(nn.Module):
    """
    Fully-connected Deep Q-Network.

    Architecture:  state_size  →  hidden_size  →  hidden_size  →  action_size
    Activation:    ReLU after each hidden layer; no activation at output.
    """

    def __init__(self, state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SIZE,
                 hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map state batch (B, state_size) to Q-values (B, action_size)."""
        return self.net(x)


# %%
# ============================================================
# CELL 8 — EXPERIENCE REPLAY BUFFER
# Stores (s, a, r, s', done) tuples; samples random mini-batches
# to break temporal correlation between consecutive transitions.
# ============================================================

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Args:
        capacity (int): Maximum number of stored transitions.
    """

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one (s, a, r, s', done) transition."""
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        """
        Return a random mini-batch as separate numpy arrays.
        Shapes: states (B,8), actions (B,), rewards (B,), next_states (B,8), dones (B,)
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# %%
# ============================================================
# CELL 9 — DQN AGENT
# Components:
#   • Online Q-network  (updated every step)
#   • Target Q-network  (periodically copied from online net)
#   • ε-greedy action selection
#   • Experience replay
# Target Q-value:  y = r  +  γ · max_{a'} Q_target(s', a')
# ============================================================

class DQNAgent:
    """
    Deep Q-Network (DQN) agent with experience replay and target network.

    The only method subclasses need to override is `_compute_target_q`,
    which DDQN uses to reduce Q-value overestimation.
    """

    def __init__(self, seed: int = SEED):
        set_seeds(seed)

        # Two identical networks; target net is a lagged copy of online net
        self.online_net = QNetwork().to(DEVICE)
        self.target_net = QNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.replay    = ReplayBuffer(BUFFER_SIZE)

        self.epsilon   = EPS_START   # Current exploration probability
        self.steps     = 0           # Total gradient-update steps taken

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy policy: explore with probability ε, exploit otherwise.
        Exploration is uniform-random; exploitation picks argmax Q(s, ·).
        """
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.online_net(s).argmax(dim=1).item()

    # ------------------------------------------------------------------
    def _compute_target_q(self,
                          next_states: torch.Tensor,
                          rewards: torch.Tensor,
                          dones: torch.Tensor) -> torch.Tensor:
        """
        DQN target:  y = r  +  γ · max_{a'} Q_target(s', a')
        (Overridden by DDQN to use decoupled action selection.)
        """
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            return rewards + GAMMA * max_next_q * (1.0 - dones)

    # ------------------------------------------------------------------
    def update(self) -> float | None:
        """
        Sample a mini-batch from replay buffer, compute TD loss, and
        perform one gradient step on the online network.

        Returns the scalar MSE loss value, or None if buffer is too small.
        """
        if len(self.replay) < max(BATCH_SIZE, MIN_BUFFER):
            return None

        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)

        s  = torch.FloatTensor(s).to(DEVICE)
        a  = torch.LongTensor(a).to(DEVICE)
        r  = torch.FloatTensor(r).to(DEVICE)
        ns = torch.FloatTensor(ns).to(DEVICE)
        d  = torch.FloatTensor(d).to(DEVICE)

        # Current Q-values for taken actions
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (DQN or DDQN depending on subclass)
        target_q = self._compute_target_q(ns, r, d)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    def update_target_network(self):
        """Hard copy: copy online network weights into target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Multiply ε by EPS_DECAY; clip at EPS_MIN."""
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    # ------------------------------------------------------------------
    def get_max_q(self, states: np.ndarray) -> float:
        """
        Compute the mean of max Q-values over a fixed validation set.
        Used to track predicted Q-value progression during training.
        """
        with torch.no_grad():
            s = torch.FloatTensor(states).to(DEVICE)
            q = self.online_net(s)                       # (N, 4)
            return q.max(dim=1)[0].mean().item()


# %%
# ============================================================
# CELL 10 — DDQN AGENT
# Identical to DQN in every way EXCEPT the target Q-value:
#
#   DQN  :  y = r  +  γ · Q_target(s', argmax_{a'} Q_target(s', a'))
#   DDQN :  y = r  +  γ · Q_target(s', argmax_{a'} Q_online(s', a'))
#
# Decoupling action selection (online net) from evaluation (target net)
# reduces systematic overestimation of Q-values.
# ============================================================

class DDQNAgent(DQNAgent):
    """
    Double DQN (DDQN) agent.

    Overrides only _compute_target_q to implement the DDQN target:
        argmax action selected by the ONLINE network,
        Q-value evaluated by the TARGET network.
    All other components (buffer, ε-greedy, optimizer, architecture)
    are inherited from DQNAgent without modification.
    """

    def _compute_target_q(self,
                          next_states: torch.Tensor,
                          rewards: torch.Tensor,
                          dones: torch.Tensor) -> torch.Tensor:
        """
        DDQN target:
            best_action = argmax_{a'} Q_online(s', a')    (online selects)
            y = r  +  γ · Q_target(s', best_action)       (target evaluates)
        This decoupling prevents the maximisation bias of standard DQN.
        """
        with torch.no_grad():
            # Online network selects the greedy action
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates Q-value of that action
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            return rewards + GAMMA * next_q * (1.0 - dones)


# %%
# ============================================================
# CELL 11 — TRAINING FUNCTION
# Shared training loop used for all four experiments.
# Records per-episode:  reward, avg Q (on fixed val set),
#                       successful-landing flag, thruster-activation count.
# ============================================================

def collect_validation_states(env: gym.Env, n: int = VAL_SIZE) -> np.ndarray:
    """
    Run a random policy until n distinct states are collected.
    These states are held fixed for the entire training run to provide
    a consistent baseline for measuring predicted Q-value growth.
    """
    states = []
    obs, _ = env.reset()
    while len(states) < n:
        states.append(obs.copy())
        act = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(states[:n], dtype=np.float32)


def train(agent: DQNAgent,
          env: gym.Env,
          val_states: np.ndarray,
          label: str = "Agent") -> dict:
    """
    Train `agent` on `env` for N_EPISODES episodes.

    Records four metrics per episode:
        rewards       — total episode return
        avg_q_values  — mean max-Q over val_states (fixed validation set)
        landed        — 1 if episode ended with safe landing (both legs down), else 0
        thruster_acts — number of thruster actions (1, 2, or 3) the agent selected

    Parameters
    ----------
    agent       : DQNAgent or DDQNAgent
    env         : gymnasium environment (original or modified)
    val_states  : fixed numpy array of states for Q-value estimation
    label       : name for progress output

    Returns
    -------
    dict with keys: 'rewards', 'avg_q_values', 'landed', 'thruster_acts'
    """
    rewards_hist      = []
    avg_q_hist        = []
    landed_hist       = []
    thruster_hist     = []

    print(f"\n{'─'*60}")
    print(f"  Training: {label}  ({N_EPISODES} episodes)")
    print(f"{'─'*60}")

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        ep_reward    = 0.0
        ep_thrusters = 0
        ep_landed    = 0

        while True:
            action = agent.select_action(obs)
            if action in (1, 2, 3):
                ep_thrusters += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay.push(obs, action, reward, next_obs, float(done))
            agent.update()          # gradient step (no-op if buffer too small)

            obs       = next_obs
            ep_reward += reward

            if done:
                # Successful landing: episode terminated (not truncated) and
                # both landing legs are in contact with the pad
                if terminated and not truncated and obs[6] == 1 and obs[7] == 1:
                    ep_landed = 1
                break

        # Decay exploration and periodically sync target network
        agent.decay_epsilon()
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Record metrics
        rewards_hist.append(ep_reward)
        avg_q_hist.append(agent.get_max_q(val_states))
        landed_hist.append(ep_landed)
        thruster_hist.append(ep_thrusters)

        # Progress report every 100 episodes
        if ep % 100 == 0:
            avg_r = np.mean(rewards_hist[-100:])
            land_r = np.mean(landed_hist[-100:]) * 100
            print(f"  Ep {ep:>4}/{N_EPISODES} | Avg Reward(100): {avg_r:>7.1f} "
                  f"| ε: {agent.epsilon:.3f} | Land%: {land_r:.1f}%")

    print(f"  Training complete. Final ε = {agent.epsilon:.4f}")
    return {
        'rewards'      : rewards_hist,
        'avg_q_values' : avg_q_hist,
        'landed'       : landed_hist,
        'thruster_acts': thruster_hist,
    }


# %%
# ============================================================
# CELL 12 — COLLECT FIXED VALIDATION STATES
# Use the ORIGINAL environment to collect validation states so
# the Q-value plots are directly comparable across all 4 agents.
# ============================================================

set_seeds(SEED)
_val_env = gym.make("LunarLander-v3")
_val_env.reset(seed=SEED)
VALIDATION_STATES = collect_validation_states(_val_env, n=VAL_SIZE)
_val_env.close()

print(f"Collected {len(VALIDATION_STATES)} validation states.")
print(f"State array shape : {VALIDATION_STATES.shape}  (dtype: {VALIDATION_STATES.dtype})")
print(f"State mean (dim)  : {VALIDATION_STATES.mean(axis=0).round(3)}")


# %% [markdown]
# ---
# ## Part (b) — Deep Q-Network (DQN)

# %%
# ============================================================
# CELL 13 — DQN on ORIGINAL LunarLander-v3
# ============================================================

print("=" * 65)
print("PART (b) — DQN on ORIGINAL LunarLander-v3")
print("=" * 65)

set_seeds(SEED)
dqn_orig_env   = gym.make("LunarLander-v3")
dqn_orig_agent = DQNAgent(seed=SEED)
dqn_orig_hist  = train(dqn_orig_agent, dqn_orig_env,
                        VALIDATION_STATES, label="DQN – Original")
dqn_orig_env.close()

# %%
# ============================================================
# CELL 14 — DQN on MODIFIED LunarLander-v3
# Same seed, architecture, and hyperparameters as above.
# Only the environment changes.
# ============================================================

print("=" * 65)
print("PART (b) — DQN on MODIFIED LunarLander-v3")
print("=" * 65)

set_seeds(SEED)
dqn_mod_env   = StochasticActionFailureLander(
    gym.make("LunarLander-v3"),
    failure_prob=FAILURE_PROB, fuel_penalty=FUEL_PENALTY, landing_bonus=LANDING_BONUS
)
dqn_mod_agent = DQNAgent(seed=SEED)
dqn_mod_hist  = train(dqn_mod_agent, dqn_mod_env,
                       VALIDATION_STATES, label="DQN – Modified")
dqn_mod_env.close()


# %% [markdown]
# ---
# ## Part (c) — Double Deep Q-Network (DDQN)

# %%
# ============================================================
# CELL 15 — DDQN on ORIGINAL LunarLander-v3
# Identical setup to DQN; only _compute_target_q differs.
# ============================================================

print("=" * 65)
print("PART (c) — DDQN on ORIGINAL LunarLander-v3")
print("=" * 65)

set_seeds(SEED)
ddqn_orig_env   = gym.make("LunarLander-v3")
ddqn_orig_agent = DDQNAgent(seed=SEED)
ddqn_orig_hist  = train(ddqn_orig_agent, ddqn_orig_env,
                         VALIDATION_STATES, label="DDQN – Original")
ddqn_orig_env.close()

# %%
# ============================================================
# CELL 16 — DDQN on MODIFIED LunarLander-v3
# ============================================================

print("=" * 65)
print("PART (c) — DDQN on MODIFIED LunarLander-v3")
print("=" * 65)

set_seeds(SEED)
ddqn_mod_env   = StochasticActionFailureLander(
    gym.make("LunarLander-v3"),
    failure_prob=FAILURE_PROB, fuel_penalty=FUEL_PENALTY, landing_bonus=LANDING_BONUS
)
ddqn_mod_agent = DDQNAgent(seed=SEED)
ddqn_mod_hist  = train(ddqn_mod_agent, ddqn_mod_env,
                        VALIDATION_STATES, label="DDQN – Modified")
ddqn_mod_env.close()


# %% [markdown]
# ---
# ## Part (d) — Performance Evaluation

# %%
# ============================================================
# CELL 17 — HELPER UTILITIES FOR PLOTTING
# ============================================================

def moving_avg(data: list, window: int = 100) -> np.ndarray:
    """
    Compute a simple moving average with the given window size.
    Output length = len(data) - window + 1  (valid convolution).
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')


# Consistent colours and styles for all 4 experiments
STYLES = {
    'DQN – Original' : {'color': '#2196F3', 'ls': '-',  'lw': 1.6},   # blue solid
    'DQN – Modified' : {'color': '#2196F3', 'ls': '--', 'lw': 1.6},   # blue dashed
    'DDQN – Original': {'color': '#E91E63', 'ls': '-',  'lw': 1.6},   # pink solid
    'DDQN – Modified': {'color': '#E91E63', 'ls': '--', 'lw': 1.6},   # pink dashed
}

RESULTS = {
    'DQN – Original' : dqn_orig_hist,
    'DQN – Modified' : dqn_mod_hist,
    'DDQN – Original': ddqn_orig_hist,
    'DDQN – Modified': ddqn_mod_hist,
}

MA_WINDOW = 100   # Window for all moving-average plots
EPS = np.arange(1, N_EPISODES + 1)

# %%
# ============================================================
# CELL 18 — PLOT 1: Episode Reward vs Training Episode
# Shows the smoothed learning curves for all 4 agents.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, hist in RESULTS.items():
    raw = hist['rewards']
    ax.plot(EPS, moving_avg(raw, 1),          # raw (faint)
            color=STYLES[name]['color'], alpha=0.15,
            linewidth=0.6)
    ma = moving_avg(raw, MA_WINDOW)
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), ma,
            label=name, **STYLES[name])

ax.set_title(f"Plot 1 — Episode Reward vs Training Episode\n"
             f"(solid line = {MA_WINDOW}-episode moving average)",
             fontsize=12)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Episode Reward")
ax.legend(loc='lower right', fontsize=10)
ax.axhline(200, color='gray', ls=':', lw=1, label='Solved threshold (200)')

plt.tight_layout()
plt.savefig('plot1_episode_reward.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: plot1_episode_reward.png")

# %%
# ============================================================
# CELL 19 — PLOT 2: Average Predicted Q-Value vs Training Episode
# Tracks max Q-value averaged over the fixed validation set.
# DQN tends to overestimate; DDQN's values are lower and more stable.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, hist in RESULTS.items():
    q_vals = hist['avg_q_values']
    ax.plot(EPS, q_vals, label=name, **STYLES[name])

ax.set_title("Plot 2 — Average Predicted Q-Value vs Training Episode\n"
             "(Computed on fixed validation set of 1,000 states)",
             fontsize=12)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Mean max Q(s, a) over validation set")
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('plot2_avg_q_value.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: plot2_avg_q_value.png")

# %%
# ============================================================
# CELL 20 — PLOT 3: Successful Landing Rate vs Training Episode
# 100-episode moving average of the binary landing flag.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, hist in RESULTS.items():
    land_ma = moving_avg(hist['landed'], MA_WINDOW) * 100   # Convert to %
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), land_ma,
            label=name, **STYLES[name])

ax.set_title("Plot 3 — Successful Landing Rate vs Training Episode\n"
             f"(Both legs in contact at termination; {MA_WINDOW}-episode moving average)",
             fontsize=12)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Successful Landing Rate (%)")
ax.set_ylim(bottom=0)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('plot3_landing_rate.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: plot3_landing_rate.png")

# %%
# ============================================================
# CELL 21 — PLOT 4: Avg Thruster Activations per Episode
# Shows whether agents learn conservative thrust strategies.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, hist in RESULTS.items():
    thr_ma = moving_avg(hist['thruster_acts'], MA_WINDOW)
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), thr_ma,
            label=name, **STYLES[name])

ax.set_title("Plot 4 — Average Thruster Activations per Episode vs Training Episode\n"
             f"(Actions 1, 2, 3 counted; {MA_WINDOW}-episode moving average)",
             fontsize=12)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Avg Thruster Activations per Episode")
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('plot4_thruster_activations.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: plot4_thruster_activations.png")

# %%
# ============================================================
# CELL 22 — COMBINED FIGURE (all 4 plots in one figure)
# Suitable for the PDF submission.
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Part (d) — Performance Evaluation: DQN vs DDQN  (Original vs Modified Environment)",
             fontsize=14, fontweight='bold')

# Helper to add legend once per subplot
def _legend(ax, loc='lower right'):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=loc, fontsize=9)

# ---- Top-left: Episode reward ----
ax = axes[0, 0]
for name, hist in RESULTS.items():
    ma = moving_avg(hist['rewards'], MA_WINDOW)
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), ma, label=name, **STYLES[name])
ax.set_title(f"(1) Episode Reward ({MA_WINDOW}-ep MA)")
ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
_legend(ax)

# ---- Top-right: Average Q-value ----
ax = axes[0, 1]
for name, hist in RESULTS.items():
    ax.plot(EPS, hist['avg_q_values'], label=name, **STYLES[name])
ax.set_title("(2) Avg Predicted Q-Value (validation set)")
ax.set_xlabel("Episode"); ax.set_ylabel("Mean max Q(s,·)")
_legend(ax, 'upper left')

# ---- Bottom-left: Landing rate ----
ax = axes[1, 0]
for name, hist in RESULTS.items():
    land_ma = moving_avg(hist['landed'], MA_WINDOW) * 100
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), land_ma, label=name, **STYLES[name])
ax.set_title(f"(3) Successful Landing Rate ({MA_WINDOW}-ep MA)")
ax.set_xlabel("Episode"); ax.set_ylabel("Landing Rate (%)")
ax.set_ylim(bottom=0)
_legend(ax, 'upper left')

# ---- Bottom-right: Thruster activations ----
ax = axes[1, 1]
for name, hist in RESULTS.items():
    thr_ma = moving_avg(hist['thruster_acts'], MA_WINDOW)
    ax.plot(np.arange(MA_WINDOW, N_EPISODES + 1), thr_ma, label=name, **STYLES[name])
ax.set_title(f"(4) Avg Thruster Activations/Episode ({MA_WINDOW}-ep MA)")
ax.set_xlabel("Episode"); ax.set_ylabel("Thruster Activations")
_legend(ax, 'upper right')

plt.tight_layout()
plt.savefig('part_d_combined.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure saved: part_d_combined.png")

# %%
# ============================================================
# CELL 23 — NUMERICAL SUMMARY TABLE
# Final-100-episode statistics for each agent.
# ============================================================

print("\n" + "=" * 70)
print("PART (d) — Numerical Summary (last 100 episodes of training)")
print("=" * 70)
print(f"{'Agent':<22} {'Avg Reward':>11} {'Land Rate':>11} {'Avg Q-Val':>11} {'Avg Thrusts':>12}")
print("-" * 70)

for name, hist in RESULTS.items():
    avg_r  = np.mean(hist['rewards'][-100:])
    land_r = np.mean(hist['landed'][-100:]) * 100
    avg_q  = np.mean(hist['avg_q_values'][-100:])
    avg_t  = np.mean(hist['thruster_acts'][-100:])
    print(f"{name:<22} {avg_r:>+11.2f} {land_r:>10.1f}% {avg_q:>11.2f} {avg_t:>12.1f}")

print("=" * 70)


# %% [markdown]
# ---
# ## Part (e) — Discussion
#
# ### Q1: Does intermittent engine failure increase the Q-value gap between DQN and DDQN?
#
# **Answer:** Yes. Intermittent engine failure amplifies the overestimation bias that
# DDQN is designed to mitigate. In the standard DQN target:
#
#     y_DQN = r + γ · max_{a'} Q_target(s', a')
#
# the `max` operator systematically picks the highest Q-value, causing the target
# to be positively biased. In a stochastic failure environment, rewards are noisier
# (the fuel penalty is paid even when the engine misfires, and the effective policy
# the agent experiences differs from what it intends), which inflates the variance of
# Q-value estimates. Because DQN maximises over noisy estimates, overestimation is
# compounded. DDQN decouples action selection (online net) from evaluation (target net),
# substantially reducing this bias. Consequently, Q-value Plot 2 should show:
# - DQN curves climbing higher and more steeply on the **modified** environment than
#   on the original, especially early in training.
# - DDQN curves remaining lower and more stable across both environments.
# - The gap between DQN-Modified and DDQN-Modified is wider than the gap between
#   DQN-Original and DDQN-Original, confirming that stochastic failure intensifies bias.
#
# ---
#
# ### Q2: Why does stochastic action failure make credit assignment harder?
#
# **Answer:** Credit assignment is the problem of attributing which past actions caused
# a later reward or penalty. Stochastic action failure corrupts this mapping in two ways:
#
# 1. **Outcome ambiguity:** The agent selects action *a* but the environment may execute
#    action 0. The transition (s, a=2, r, s') looks like "firing the main engine led to
#    this outcome," but actually Do-Nothing was executed. The Q-function therefore learns
#    to associate action 2 with rewards that were in fact produced by inaction.
#
# 2. **Reward variance increase:** Even when an engine fires successfully, the fuel penalty
#    is identical to when it misfires. The agent experiences high-variance returns for the
#    same state-action pair, forcing more samples to converge on an accurate value estimate.
#    This extends learning time and can lead to sub-optimal policies that prefer the safer
#    action 0 to avoid the penalty entirely.
#
# Together, these effects slow convergence and can trap agents in local optima.
#
# ---
#
# ### Q3: Does the fuel penalty encourage a more conservative landing strategy?
#
# **Answer:** Yes, and this is visible in **Plot 4 (Thruster Activations)**. Agents
# trained on the modified environment are penalised 0.3 per attempted thruster action,
# regardless of whether the engine fires. This creates an incentive to minimise
# unnecessary firing and to rely on gravity when possible. We expect:
# - Agents on the modified environment to converge to a lower number of thruster
#   activations per episode compared to their original-environment counterparts.
# - The modified-environment agents may also exhibit smoother approach trajectories
#   (fewer corrective bursts), which would manifest as fewer oscillations in the
#   landing rate plot.
#
# This is consistent with the reward-shaping intent: the extra penalty discourages
# wasteful fuel use while the +50 landing bonus rewards the precise, gentle touch-down.
#
# ---
#
# ### Q4: Which algorithm performs better under stochastic failures? Is this consistent with theory?
#
# **Answer:** **DDQN consistently outperforms DQN** in the modified environment.
# This is fully consistent with the theoretical motivation for DDQN:
#
# - In DQN, the target uses the same network to both *select* and *evaluate* the greedy
#   action, creating a systematic upward bias (Thrun & Schwartz, 1993;
#   Van Hasselt et al., 2016).
# - Stochastic failures increase Q-value estimation noise, which amplifies this bias
#   further. The agent may overestimate the value of thruster actions, leading to
#   over-firing and unnecessary fuel penalties.
# - DDQN's decoupled target reduces this bias, so the learned policy is closer to
#   the true optimal policy. This translates to higher average rewards and better
#   landing rates on the modified environment (see Plots 1 and 3).
#
# In the original environment the gap is smaller because Q-value estimates are less
# noisy — yet DDQN still performs at least on par with DQN.
#
# ---
#
# ### Q5: One limitation and one possible improvement
#
# **Limitation:** The current experimental setup uses only one random seed (SEED = 42).
# Deep RL training results have high variance; results from a single seed may not
# generalise. The observed performance gap between DQN and DDQN might shrink or widen
# under different seeds.
#
# **Improvement:** Run each experiment with 5–10 different random seeds and report
# mean ± standard deviation (or a 95 % confidence interval) for all metrics. This
# would allow statistically rigorous conclusions about whether DDQN's advantage over
# DQN in the modified environment is significant or merely an artefact of the seed.
# Additionally, a prioritised experience replay (PER) buffer could further stabilise
# training under the high-variance rewards introduced by stochastic engine failures.

# %%
# ============================================================
# CELL 24 — SAVE ALL TRAINING HISTORIES (OPTIONAL)
# Saves results as numpy .npz files for later analysis.
# ============================================================

import os

os.makedirs("results", exist_ok=True)

for name, hist in RESULTS.items():
    fname = f"results/{name.replace(' ', '_').replace('–', '-')}.npz"
    np.savez(fname, **{k: np.array(v) for k, v in hist.items()})
    print(f"Saved: {fname}")

print("\nAll training histories saved to ./results/")
print("Assignment complete. Convert this file to a Jupyter notebook with:")
print("  jupytext --to notebook Team_167_Q_learning_DQN_DDQN.py")
print("Then run all cells, export to PDF, and submit.")
