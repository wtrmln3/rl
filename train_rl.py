from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from dynamics import MassDamperEnv

env = MassDamperEnv()
check_env(env)  

model = SAC('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=50000)  

model.save("mass_damper_sac")
print("RL model saved!")
