# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch.nn.functional as F
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, state_size, action_size, num_agents,batchsize,discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents),
                             DDPGAgent(state_size, action_size, num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.num_agents = num_agents 
        self.batchsize = batchsize
    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for i in range(self.num_agents):
            action = self.maddpg_agent[i].act(obs_all_agents[i, :].view(1, -1), noise)
            actions.append(action.squeeze()) 
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        obs_all_agents = torch.tensor(obs_all_agents, dtype=torch.float)
        for i in range(self.num_agents):
            action = self.maddpg_agent[i].target_act(obs_all_agents[:, i, :],  noise) 
            target_actions.append(action) 
        return target_actions
    
    def update(self, samples, agent_number):
        state, full_state, action, reward, next_state, full_next_state, done = samples
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_state.view(self.batchsize, self.num_agents, -1))
        target_actions = torch.cat(target_actions, dim=1)
        
        with torch.no_grad():
            q_next = agent.target_critic(full_next_state, target_actions.to(device))
       
        y = reward[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:, agent_number].view(-1, 1))

        q = agent.critic(full_state, action.view(self.batchsize, -1))
        
        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(q, y.detach())
        
        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        q_input = [
            self.maddpg_agent[i].actor(state.view([self.batchsize, self.num_agents, -1])[:, i, :]) if i == agent_number else
            self.maddpg_agent[i].actor(state.view([self.batchsize, self.num_agents, -1])[:, i, :]).detach() for i in
            range(self.num_agents)]
                
        q_input = torch.cat(q_input, dim=1)
        
        
        # get the policy gradient
        actor_loss = -agent.critic(full_state,q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)
        agent.actor_optimizer.step()

        self.update_targets()
        
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




