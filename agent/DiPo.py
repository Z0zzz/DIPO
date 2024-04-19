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
            self.actor = Diffusion(args, state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps).to(device)
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
        print("dipo device: ", self.device)
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

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        pred_actions, action = self.actor(state, eval).cpu().data.numpy().flatten()
        pred_actions = pred_actions.clip(-1,1)
        action = action.clip(-1, 1)
        pred_actions = pred_actions * self.action_scale + self.action_bias
        action = action * self.action_scale + self.action_bias
        return pred_actions, action

    def action_gradient(self, batch_size, log_writer):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        
        action_horizon_best_actions = best_actions.detach().clone()[:, self.act_horizon_start:self.act_horizon_end]
        # DIPO: reshape to pass to critic
        actions_optim = torch.optim.Adam([action_horizon_best_actions], lr=self.action_lr, eps=1e-5)
        
        # actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            action_horizon_best_actions.requires_grad_(True)

            # DIPO: reshape to process in critic
            states_ = torch.flatten(states, start_dim=1)
            action_horizon_best_actions_ = torch.flatten(action_horizon_best_actions, start_dim=1)

            q1, q2 = self.critic(states_, action_horizon_best_actions_)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            action_horizon_best_actions.requires_grad_(False)  
            action_horizon_best_actions.clamp_(-1., 1.)

        best_actions_new = best_actions.detach().cpu().numpy()
        best_actions_new[:, self.act_horizon_start:self.act_horizon_end] = action_horizon_best_actions.detach().cpu().numpy()
        self.diffusion_memory.replace(idxs, best_actions_new)

        return states, torch.tensor(best_actions_new).to(self.device)

    def train(self, iterations, batch_size=256, global_step = 0, log_writer=None):
        for _ in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)

            _, next_actions = self.actor_target(next_states, self.actor_target)
            print("next action: ", next_actions.shape)
            next_states_flatten = torch.flatten(next_states, start_dim=1)
            
            next_actions = torch.flatten(next_actions, start_dim=1)
            print("next action: ", next_actions.shape)
            
            target_q1, target_q2 = self.critic_target(next_states_flatten, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()
            
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)
            critic_loss = q1_loss + q2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.actor.loss(best_actions, states)

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
        log_writer.add_scalar("losses/qf1_values", current_q1.mean().item(), global_step)
        log_writer.add_scalar("losses/qf2_values", current_q2.mean().item(), global_step)
        log_writer.add_scalar("losses/qf1_loss", q1_loss.item(), global_step)
        log_writer.add_scalar("losses/qf2_loss", q2_loss.item(), global_step)
        log_writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        log_writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

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

