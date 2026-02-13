import torch
import numpy as np
from models.explainer import MolDQN
from models.explainer.ReplayMemory import ReplayMemory

class Agent(object):
    def __init__(self,
                 num_input,
                 num_output,
                 device,
                 lr,
                 replay_buffer_size
    ):

        self.device = device
        self.num_input = num_input
        self.num_output = num_output

        self.dqn, self.target_dqn = (
            MolDQN(num_input, num_output).to(self.device),
            MolDQN(num_input, num_output).to(self.device)
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(replay_buffer_size)
        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(), lr=lr
        )

    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()

        return action

    def train_step(self, batch_size, gamma, polyak):

        if len(self.replay_buffer) > 0:
            t = self.replay_buffer.memory[0]
            print("transition length:", len(t))
            
            for i, x in enumerate(t):
                shp = getattr(x, "shape", None)
                print(i, type(x), shp)


        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience]).to(self.device) 
        q = self.dqn(states_).squeeze(-1)                   

        next_states_ = [S.to(self.device) for *_, S, _ in experience]

        #Double DQN target
        q_next = []
        with torch.no_grad():
            for S in next_states_:
                # online selects
                q_online_all = self.dqn(S).squeeze(-1)
                a_star = torch.argmax(q_online_all, dim=0)

                # target evaluates
                q_target_all = self.target_dqn(S).squeeze(-1)
                q_next.append(q_target_all[a_star])

        q_next = torch.stack(q_next)                 

        rewards = torch.stack([R for _, R, *_ in experience]).to(self.device)  
        rewards = rewards.view(-1)                                       

        dones = torch.tensor([D for *_, D in experience], device=self.device).float() 

        y = rewards + gamma * (1.0 - dones) * q_next                  
        td = q - y                                                             


        loss = torch.where(
            torch.abs(td) < 1.0,
            0.5 * td * td,
            1.0 * (torch.abs(td) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return loss
