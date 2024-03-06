import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.model import MLP, Critic
from agent.diffusion import Diffusion
from agent.vae import VAE
from agent.helpers import EMA
from agent.model import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class DiPo(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_space,
                 memory,
                 diffusion_memory,
                 device,
        ):
        action_dim = np.prod(action_space.shape)
        
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.act_horizon_start = args.obs_horizon - 1
        self.act_horizon_end = self.act_horizon_start + args.act_horizon
        
        self.policy_type = args.policy_type
        if self.policy_type == 'Diffusion':
            # self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
            #                        beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps).to(device)
            self.actor = ConditionalUnet1D(
                input_dim=action_dim, # act_horizon is not used (U-Net doesn't care)
                global_cond_dim=state_dim, # obs_horizon * obs_dim
                diffusion_step_embed_dim=args.diffusion_step_embed_dim,
                down_dims=args.unet_dims,
                n_groups=args.n_groups,
            ).to(device)
            
            self.num_train_diffusion_iters = args.n_timesteps
            self.num_eval_diffusion_iters = args.num_eval_diffusion_iters
            
            # DIPO: use DDIM to speedup inference
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_train_diffusion_iters,
                beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
                clip_sample=True, # clip output to [-1,1] to improve stability
                prediction_type='epsilon' # predict noise (instead of denoised action)
            )
            self.noise_scheduler.set_timesteps(self.num_eval_diffusion_iters)
        
            self.sample_eval_action = self.sample_action
        elif self.policy_type == 'VAE':
            self.actor = VAE(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        else:
            self.actor = MLP(state_dim=state_dim, action_dim=action_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim*self.act_horizon).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim

        self.action_lr = args.action_lr

        self.device = device

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def append_memory(self, state, action, reward, next_state, mask, pred_horizon_actions):
        action = (action - self.action_bias) / self.action_scale
        
        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, pred_horizon_actions)

    def sample_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            # obs_cond = torch.flatten(obs_seq) # (B, obs_horizon * obs_dim)
            obs_cond = obs_seq
            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.action_dim), device=obs_seq.device)
            
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.actor(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        # DIPO: collapse last two dimensions 
        return noisy_action_seq, noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

    def action_gradient(self, batch_size, log_writer):
        
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        action_horizon_best_actions = best_actions.detach().clone()[:, self.act_horizon_start:self.act_horizon_end]
        # DIPO: reshape to pass to critic
        actions_optim = torch.optim.Adam([action_horizon_best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            action_horizon_best_actions.requires_grad_(True)

            # DIPO: reshape to process in critic
            states_ = torch.flatten(states, start_dim=1)
            action_horizon_best_actions_ = torch.flatten(action_horizon_best_actions, start_dim=1)
            # pdb.set_trace()
            
            q1, q2 = self.critic(states_, action_horizon_best_actions_)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()
            # print("action loss: ", loss.mean())
            loss.backward(torch.ones_like(loss))
            
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()
            # pdb.set_trace()

            action_horizon_best_actions.requires_grad_(False)  
            action_horizon_best_actions.clamp_(-1., 1.)
    
        best_actions_new = best_actions.detach().cpu().numpy()
        best_actions_new[:, self.act_horizon_start:self.act_horizon_end] = action_horizon_best_actions.detach().cpu().numpy()
        self.diffusion_memory.replace(idxs, best_actions_new)

        return states, torch.tensor(best_actions_new).to(self.device)

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)
            
            print("q values: ", current_q1.mean().item(), " ", current_q2.mean().item())
            
            next_pred_actions, next_actions = self.sample_action(next_states)
            next_states_flatten = torch.flatten(next_states, start_dim=1)
            next_actions = torch.flatten(next_actions, start_dim=1)

            target_q1, target_q2 = self.critic_target(next_states_flatten, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            print("q loss: ", critic_loss.item())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()
            print("optimizer step")
            import pdb
            # pdb.set_trace()
            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.compute_loss(states, best_actions, self.device)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
            self.actor_optimizer.step()

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.step += 1

    def compute_loss(self, obs_seq, action_seq, device):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
        # print("action dim: ", action_seq.shape)
        # sample noise to add to actions
        # DIPO: there is no prediction horizon? 
        noise = torch.randn((B, self.pred_horizon, self.action_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)
        
        # predict the noise residual
        noise_pred = self.actor(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)
    
    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

