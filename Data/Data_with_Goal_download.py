
 
import os
import gym
import numpy as np
 
import collections
import pickle
 
import d4rl
 
datasets = []
 
data_dir = "./data"
 
print(data_dir)
 
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for env_name in ['antmaze']:
    for dataset_type in ['umaze','medium']:

        name = f'{env_name}-{dataset_type}-v1'
        name1 = f'{env_name}-{dataset_type}-goal'
        pkl_file_path = os.path.join(data_dir, name1)
 
        print("processing: ", name)
 
        env = gym.make(name)
        dataset = env.get_dataset()
 
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)
 
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True
 
        episode_step = 0
        paths = []
        episode_goals = [] 
        for i in range(N):

            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'actions', 'rewards', 'terminals','infos/goal']:
                data_[k].append(dataset[k][i])

            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                if len(episode_data['rewards']) > 1: ##########Only save trajectories with length >1
                    #print(episode_data['infos/goal'])
                    
                    merged_value = np.concatenate([episode_data['observations'], episode_data['infos/goal']], axis=1)
                    episode_data['observations'] = merged_value
                    episode_goals.append(episode_data['infos/goal'][0])
                    del episode_data['infos/goal']
                    paths.append(episode_data)
  
                data_ = collections.defaultdict(list)
            episode_step += 1
 
        returns = np.array([np.sum(p['rewards']) for p in paths])
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
 
        with open(f'{pkl_file_path}.pkl', 'wb') as f:
            pickle.dump(paths, f)
