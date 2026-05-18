import gym
import d4rl # Import required to register environments
import mjrl
import numpy as np
def check_d4rl():
    #Create the environment
    env = gym.make('halfcheetah-medium-v2')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations']) # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    dataset = d4rl.qlearning_dataset(env)


def check_generate_dataset(dataset_name='halfcheetah-medium-v2'):
    #/data/yuzhengyang/RL_Learning/EasyRL4Rec/data/data/generated_data/halfcheetah-medium-v2/gta_smaples.npz
    file_path = f"./data/data/generated_data/{dataset_name}/gta_smaples.npz"
    data = np.load(file_path, allow_pickle=True) 
    config_dict = data['config'].item() # 这里的config是gta.yaml的配置文件
    data = data['data'].squeeze()
  #  print(data)
    
    print(data[0].keys())

if __name__ == "__main__":
    #check_d4rl()
    check_generate_dataset()
