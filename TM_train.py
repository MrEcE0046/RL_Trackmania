from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from agent import TrackmaniaEnv

# Skapa miljö
env = TrackmaniaEnv()

# Valfritt: kontrollera att miljön uppfyller Gymnasium-kraven
check_env(env, warn=True)

# Skapa modell
model = DQN(
    "MlpPolicy", env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
)

# Träna modellen
model.learn(total_timesteps=50000)

# Spara modell
model.save("dqn_trackmania")