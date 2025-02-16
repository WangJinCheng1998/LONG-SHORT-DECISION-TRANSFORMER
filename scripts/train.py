import argparse #命令行库
import os #系统库
import sys
import random
import csv
from datetime import datetime
print(sys.path)
import numpy as np
import gym
import torch.distributions as D
import d4rl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from decision_transformer.utils_o import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score
from decision_transformer.LSDT import DecisionTransformer


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_combined_lambda(warmup_steps, total_updates):

    def combined_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 0.5 * (1 + np.cos((step - warmup_steps) / (total_updates - warmup_steps) * np.pi))
    
    return combined_lambda

def train(args):
    set_seeds(args.seed)
    print(np.random.randn(5))
    
    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go
    kernelsize= args.kernel_size
    warmup_steps = args.warmup_steps 
    index=False
    if args.env == 'walker2d':
        env_name = 'Walker2d-v2'
        rtg_target = 7000
        env_d4rl_name = f'walker2d-{dataset}-v2'
    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v2'
        rtg_target = 7000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v2'
        rtg_target = 7000
        env_d4rl_name = f'hopper-{dataset}-v2'

    elif   args.env == 'maze2d':
        env_name = f'maze2d-{dataset}-v1'
        rtg_target = 300
        if args.goalconcate:
            env_d4rl_name = f'maze2d-{dataset}-v6' # we denote our modified dataset as v6
            index=True
        else:
            env_d4rl_name = f'maze2d-{dataset}-v2'
    elif args.env == 'antmaze':
        env_name = f'antmaze-{dataset}-v2'
        rtg_target =100
        if args.goalconcate:
            env_d4rl_name = f'antmaze-{dataset}-v6' # we denote our modified dataset as v6
        else:
            env_d4rl_name = f'antmaze-{dataset}-v2'

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
   

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = str(lr) +"_" + str(context_len)+"_" +  str(wt_decay)+"_" +str(n_blocks)+"_" +str(kernelsize)

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)



    traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale) 

    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                         
                        )
    
    data_iter = iter(traj_data_loader)

    ## get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()
    env = gym.make(env_name)

    if args.goalconcate:
        state_dim=env.observation_space.shape[0]+2 # 2 correspoding with the dimension of Goal state
    else:
        state_dim = env.observation_space.shape[0]

    act_dim = env.action_space.shape[0]
    convdim=args.convdim
    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
                kernelsize=kernelsize, # Convolution kernersize
                convdim=convdim,
            ).to(device)

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    total_updates=num_updates_per_iter*max_train_iters
    lr_lambda = create_combined_lambda(warmup_steps, total_updates)

    # 创建 scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    max_d4rl_score = -1.0
    total_updates = 0

    print('target',index)

    for i_train_iter in range(max_train_iters):
        TT=0
        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask= next(data_iter)
   
            timesteps, states, actions,returns_to_go, traj_mask = timesteps.to(device), states.to(device), actions.to(device),returns_to_go.to(device).unsqueeze(dim=-1),  traj_mask.to(device)

            action_target = torch.clone(actions).detach().to(device)
  

            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                          
                                                        )
  
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]
  

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')


            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())
        render=False
        results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len,args.goalconcate, state_mean, state_std,render, index
                                )

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']

        eval_d4rl_score = get_d4rl_normalized_score(eval_avg_reward, env_name,index) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                "eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )

        print(log_str)
 

        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    eval_d4rl_score]
        csv_writer.writerow(log_data)
 
    
        # save model
        print(f"learning rate: {lr} context_len: {context_len} weight decay: {wt_decay} kernelsize: {kernelsize} Dropout{dropout_p}")

        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)
    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=100)
    parser.add_argument('--dataset_dir', type=str, default='/home/data/')
    parser.add_argument('--log_dir', type=str, default='AntMaze/')
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--dropout_p', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=500)
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--convdim', type=str, default=64)
    parser.add_argument('--goalconcate', action='store_true', help='True for goal concate') #To indicate whether use the goal-state concatenation

    args = parser.parse_args()
    train(args) 