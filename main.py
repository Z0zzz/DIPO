import argparse
import numpy as np
import torch

from agent.DiPo import DiPo
from agent.replay_memory import ReplayMemory, DiffusionMemory
import datetime
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import os


def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=100, metavar='N',
                        help='diffusion timesteps (default: 100)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                        help='diffusion learning rate (default: 0.0003)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')
    parser.add_argument("--wandb-project-name", type=str, default="ManiSkill2-dev",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    
    parser.add_argument("--obs-horizon", type=int, default=1)
    # Seems not very important in ManiSkill, 1, 2, 4 work well
    parser.add_argument("--act-horizon", type=int, default=1)
    # Seems not very important in ManiSkill, 4, 8, 15 work well
    parser.add_argument("--pred-horizon", type=int, default=16)
    # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64) # not very important
    parser.add_argument("--unet-dims", metavar='N', type=int, nargs='+', default=[64, 128, 256]) # ~4.5M params
    parser.add_argument("--n-groups", type=int, default=8)
    
    return parser.parse_args()

    

def evaluate(env, agent, writer, steps, device):
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        truncated = False
        count = 0
        print("episode: ", i, flush=True)
        while not (done or truncated):
            print("evaluation device: ", device)
            pred_actions, action = agent.sample_action(torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device))
            action = np.squeeze(action.detach().cpu().numpy(), axis=None)
            next_state, reward, done, truncated,  _ = env.step(action)
            count += 1
            episode_reward += reward
            # print("done: ", done)
            # print("truncates: ", truncated)
            
            # print("episode reward: ", episode_reward)
            # print("counting: ", count)
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)

    writer.add_scalar(
            'reward/test', mean_return, steps)
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
              f'reward: {mean_return:<5.1f}')
    print('-' * 60)


def main(args=None):
    if args is None:
        args = readParser()

    device = torch.device(int(args.cuda)) if args.cuda != "cpu" else "cpu"
    print("main device: ", device)
    ALGO_NAME="DIPO"
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(args.env_name, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
        )

    # dir = "record"
    # dir = "test"
    # log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'ratio={args.ratio}', f'seed={args.seed}')
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    # Initial environment
    env = gym.make(args.env_name)
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = 10000
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10
    
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, args.pred_horizon, memory_size, device)

    agent = DiPo(args, state_size, env.action_space, memory, diffusion_memory, device)

    steps = 0
    episodes = 0
    act_horizon_start = args.obs_horizon - 1
    act_horizon_end = act_horizon_start + args.act_horizon
    
    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        truncated = False
        state, _ = env.reset()
        episodes += 1
        while not (done or truncated):
            if start_steps > steps:
                pred_horizon_actions = np.array([env.action_space.sample() for _ in range(args.pred_horizon)])
                actions = pred_horizon_actions[act_horizon_start:act_horizon_end][0]  # execute only act_horizon actions
                actions = np.squeeze(actions, axis=None)
            else:
                pred_actions, actions = agent.sample_action(torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device))
                actions = np.squeeze(actions.detach().cpu().numpy(), axis=None)
            next_state, reward, done, truncated,  _ = env.step(actions)
            mask = 0.0 if done else args.gamma

            steps += 1
            episode_steps += 1
            episode_reward += reward

            agent.append_memory(state, actions, reward, next_state, mask, pred_horizon_actions)

            if steps >= start_steps:
                agent.train(updates_per_step, batch_size=batch_size, global_step=steps, log_writer=writer)

            if steps % eval_interval == 0:
                evaluate(env, agent, writer, steps, device)
                # self.save_models()
                done =True

            state = next_state

        if episodes % log_interval == 0:
            writer.add_scalar('reward/train', episode_reward, steps)

        print(f'episode: {episodes:<4}  '
            f'episode steps: {episode_steps:<4}  '
            f'reward: {episode_reward:<5.1f}')


if __name__ == "__main__":
    main()
