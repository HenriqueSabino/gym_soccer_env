import random

from einops import rearrange
# from gym.spaces import flatdim
from gymnasium.spaces import flatdim
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from fastmarl.utils.models import MultiAgentSEPSNetwork, MultiAgentFCNetwork
from fastmarl.utils.standarize_stream import RunningMeanStd

# from utils.models import MultiAgentSEPSNetwork, MultiAgentFCNetwork
# from utils.standarize_stream import RunningMeanStd
# import numpy as np

class QNetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        layers,
        parameter_sharing,
        use_orthogonal_init,
        device,
    ):
        super().__init__()
        hidden_size = list(layers)
        optimizer = getattr(optim, cfg.optimizer)
        lr = cfg.lr

        self.action_space = action_space

        self.n_agents = 6 # 1 # len(obs_space)
        obs_shape = [ flatdim(obs_space) ] * self.n_agents # [flatdim(o) for o in np.array(obs_space)]
        action_shape = [ action_space.n ] * self.n_agents # [flatdim(a) for a in np.array(action_space)]


        # print("@@@@@@@@@")
        # print(obs_space)
        # print(action_space)
        # print("@@@@@@@@@")
        # print(cfg)
        # print(obs_space.shape)
        # print(action_space.n)
        # print("@@@@@@@@@")
        # print(obs_shape)
        # print(obs_space.shape)
        # print(flatdim(obs_space))
        # print(action_shape)
        # print("@@@@@@@@@")

        # MultiAgentFCNetwork is much faster that MultiAgentSepsNetwork
        # We would like to keep this, so a simple `if` switch is implemented below
        if not parameter_sharing:
            # input(">>> aqui not parameter_sharing")
            self.critic = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape, use_orthogonal_init)
            self.target = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape, use_orthogonal_init)
        else:
            self.critic = MultiAgentSEPSNetwork(
                obs_shape, hidden_size, action_shape, parameter_sharing, use_orthogonal_init
            )
            self.target = MultiAgentSEPSNetwork(
                obs_shape, hidden_size, action_shape, parameter_sharing, use_orthogonal_init
            )

        self.soft_update(1.0)
        self.to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        self.optimizer = optimizer(self.critic.parameters(), lr=lr)

        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.device = device

        self.updates = 0
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.standardize_returns = cfg.standardize_returns
        self.ret_ms = RunningMeanStd(shape=(self.n_agents,))

        print(self)

    def forward(self, inputs):
        raise NotImplemented("Forward not implemented. Use act or update instead!")

    def act(self, inputs, epsilon):
        if epsilon > random.random():
            actions = self.action_space.sample()
            return actions
        with torch.no_grad():
            inputs = [torch.from_numpy(i).to(self.device) for i in inputs]
            actions = [x.argmax(-1).cpu().item() for x in self.critic(inputs)]
        return actions
    
    def _compute_loss(self, batch):
        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
        rewards = torch.stack(
            [batch["rew"][:, i].view(-1, 1) for i in range(self.n_agents)]
        )
        dones = batch["done"].unsqueeze(0).repeat(self.n_agents, 1, 1)

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(nobs))
            q_next_states = torch.stack(self.target(nobs))
        all_q_states = torch.stack(self.critic(obs))

        a_prime = q_tp1_values.argmax(-1)
        target_next_states = q_next_states.gather(-1, a_prime.unsqueeze(-1))

        target_states = rewards + self.gamma * target_next_states * (1 - dones)

        if self.standardize_returns:
            self.ret_ms.update(target_states)
            target_states = (
                target_states - self.ret_ms.mean.view(-1, 1, 1)
            ) / torch.sqrt(self.ret_ms.var.view(-1, 1, 1))

        q_states = all_q_states.gather(-1, action)
        return torch.nn.functional.mse_loss(q_states, target_states)


    def update(self, batch):
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_from_target()

    def update_from_target(self):
        if (
            self.target_update_interval_or_tau > 1.0
            and self.updates % self.target_update_interval_or_tau == 0
        ):
            # Hard update
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            # Soft update
            self.soft_update(self.target_update_interval_or_tau)
        self.updates += 1

    def soft_update(self, t):
        source, target = self.critic, self.target
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class VDNetwork(QNetwork):
    def __init__(self, obs_space, action_space, cfg, layers, parameter_sharing, use_orthogonal_init, device):
        super().__init__(obs_space, action_space, cfg, layers, parameter_sharing, use_orthogonal_init, device)
        self.ret_ms = RunningMeanStd(shape=(1,))
    
    def _compute_loss(self, batch):
        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
        rewards = batch["rew"]
        dones = batch["done"]

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(nobs))
            q_next_states = torch.stack(self.target(nobs))
        all_q_states = torch.stack(self.critic(obs))

        _, a_prime = q_tp1_values.max(-1)
        target_next_states = q_next_states.gather(2, a_prime.unsqueeze(-1)).sum(0)
        target_states = rewards + self.gamma * target_next_states * (1 - dones)

        if self.standardize_returns:
            self.ret_ms.update(target_states)
            target_states = (target_states - self.ret_ms.mean.view(-1, 1)) / torch.sqrt(
                self.ret_ms.var.view(-1, 1)
            )

        q_states = all_q_states.gather(2, action).sum(0)
        return torch.nn.functional.mse_loss(q_states, target_states)


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim, hypernet_layers, hypernet_embed):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim

        self.embed_dim = embed_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed = hypernet_embed

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        # TODO: args undefined?
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        agent_qs = rearrange(agent_qs, "N B 1 -> B 1 N")
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1)
        return q_tot


class QMixNetwork(QNetwork):
    def __init__(self, obs_space, action_space, cfg, layers, parameter_sharing, use_orthogonal_init, mixing, device):
        super().__init__(obs_space, action_space, cfg, layers, parameter_sharing, use_orthogonal_init, device)
        self.ret_ms = RunningMeanStd(shape=(1,))

        state_dim = sum([flatdim(o) for o in obs_space])
        self.mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.target_mixer = QMixer(self.n_agents, state_dim, **mixing)
        self.soft_update(1.0)

        for param in self.target_mixer.parameters():
            param.requires_grad = False

        self.optimizer = self.optimizer_class(
            list(self.critic.parameters()) + list(self.mixer.parameters()), lr=cfg.lr,
        )
        print(self)
    
    def _compute_loss(self, batch):
        obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
        rewards = batch["rew"]
        dones = batch["done"]

        with torch.no_grad():
            q_tp1_values = torch.stack(self.critic(nobs))
            q_next_states = torch.stack(self.target(nobs))
        all_q_states = torch.stack(self.critic(obs))

        _, a_prime = q_tp1_values.max(-1)

        target_next_states = self.target_mixer(
            q_next_states.gather(2, a_prime.unsqueeze(-1)), torch.concat(nobs, dim=-1)
        ).detach()

        target_states = rewards + self.gamma * target_next_states * (1 - dones)

        if self.standardize_returns:
            self.ret_ms.update(target_states)
            target_states = (target_states - self.ret_ms.mean.view(-1, 1)) / torch.sqrt(
                self.ret_ms.var.view(-1, 1)
            )

        q_states = self.mixer(all_q_states.gather(2, action), torch.concat(obs, dim=-1))
        return torch.nn.functional.mse_loss(q_states, target_states)

    def soft_update(self, t):
        super().soft_update(t)
        try:
            source, target = self.mixer, self.target_mixer
        except AttributeError: # fix for when qmix has not initialised a mixer yet
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
