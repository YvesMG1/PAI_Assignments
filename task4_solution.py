import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np

import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ActorNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and value networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(ActorNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.

        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)
        ])

        # Retrieve activation function from string
        self.activation = getattr(nn, activation)()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        s = self.activation(self.input_layer(s))
        for hidden_layer in self.hidden_layers:
            s = self.activation(hidden_layer(s))
        s = self.output_layer(s)
        clamp = nn.Tanh()
        s = clamp(s)
        # add Tanh() layer in NN (actor)
        # separate critic network
        return s
    
class CriticNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and value networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(CriticNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.

        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)
        ])

        # Retrieve activation function from string
        self.activation = getattr(nn, activation)()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        s = self.activation(self.input_layer(s))
        for hidden_layer in self.hidden_layers:
            s = self.activation(hidden_layer(s))
        s = self.output_layer(s)
        # add Tanh() layer in NN (actor)
        # separate critic network

        return s
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()
        self.optimizer=optim.Adam(list(self.actor.parameters()), lr=actor_lr)

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 

        self.actor = ActorNetwork(
            input_dim=self.state_dim
            , output_dim=self.action_dim #+1
            , hidden_size=self.hidden_size
            , hidden_layers=self.hidden_layers
            , activation="ReLU"
        )

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.

        with torch.no_grad():
            NN_output = self.actor(state)

        # Clamp outputs to [-1,1]
        clamp = nn.Tanh()

        # Single state
        if state.shape == (3,):
            mu = NN_output
            # mu = NN_output[0]
            # log_std = NN_output[1]

            #deterministic
            if deterministic:
                action = mu
                log_prob = torch.tensor([1])
            
            #random
            elif not deterministic:
                # log_std = self.clamp_log_std(log_std)
                # std = torch.exp(log_std)
                # action_dist = Normal(mu, std)
                action_dist = Normal(0,scale=0.2) # scale=1
                # action = action_dist.rsample()
                epsilon = action_dist.rsample()
                action = mu + epsilon
                action = clamp(action)
                # log_prob = action_dist.log_prob(action)
                log_prob = action_dist.log_prob(epsilon)
                log_prob = log_prob.unsqueeze(0)

            #reshape
            # action = action.unsqueeze(0)

        # batch of states
        elif state.shape[1] == self.state_dim:
            mu = NN_output
            # print("Mu has shape {}".format(mu.shape))
            # mu = NN_output[:,0]
            # log_std = NN_output[:,1]

            #deterministic
            if deterministic:
                action = mu
            
            #random
            elif not deterministic:
                # log_std = self.clamp_log_std(log_std)
                # std = torch.exp(log_std)
                # action_dist = Normal(mu, std)
                # action = action_dist.rsample()
                n = mu.size(0)
                action_dist = Normal(torch.zeros(n), 1*torch.ones(n))
                epsilon = action_dist.rsample()
                action = mu + epsilon.unsqueeze(1)

                # print("Epsilon: {}".format(epsilon.shape))

                action = clamp(action)
                log_prob = action_dist.log_prob(epsilon)
            
            #reshape
            # action = action.unsqueeze(1)
            log_prob = log_prob.unsqueeze(1)

        assert (action.shape == (self.action_dim,) and \
                log_prob.shape == (self.action_dim,)) or (action.shape == (state.shape[0], 1) and \
                log_prob.shape == (state.shape[0], 1)), "Incorrect shape for action or log_prob. Action shape: {}, log_prob shape: {}".format(action.shape, log_prob.shape)
        
        return action, log_prob

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()
        self.optimizer = optim.Adam(list(self.Q.parameters()), lr=critic_lr, weight_decay=1e-2)

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        
        self.Q = CriticNetwork(
            input_dim=self.state_dim + self.action_dim
            , output_dim=1
            , hidden_layers=self.hidden_layers
            , hidden_size=self.hidden_size
            , activation="ReLU"
        )
        pass

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.actions_sampled = 0
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.setup_agent()
        self.obs_variance = 1

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self.policy = Actor(
            hidden_size=256
            , hidden_layers=2
            , actor_lr=1e-4
            , device=self.device
        )
        self.critic = Critic(
            hidden_size=256
            , hidden_layers=2
            , critic_lr=1e-3
            , device=self.device
        )

        self.policy_target = Actor(
            hidden_size=256
            , hidden_layers=2
            , actor_lr=1e-4
            , device=self.device
        )
        self.critic_target = Critic(
            hidden_size=256
            , hidden_layers=2
            , critic_lr=1e-3
            , device=self.device
        )

        self.policy_target.actor.load_state_dict(self.policy.actor.state_dict())
        self.critic_target.Q.load_state_dict(self.critic.Q.state_dict())

        self.tau = 0.001
        self.gamma = 0.99
        self.alpha = 0.2

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action = np.random.uniform(-1, 1, (1,))

        action = self.policy.actor(torch.tensor(s))
        action = action.detach().numpy()

        self.actions_sampled +=1

        if self.actions_sampled % 500 == 0:
            self.obs_variance = max(self.obs_variance-0.05, 0.1)
        
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        
        if train:
            noise = np.random.normal(0, self.obs_variance , 1) # add random noise for exploration during training
            return np.clip(action + noise, -1, 1) # clip to domain
        
        else:
            noise = np.random.normal(0, 0.1 , 1) # add (a bit of) random noise for more stable testing
            return np.clip(action + noise, -1, 1) # clip to domain


    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net, target_net, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        tau = self.tau
        gamma = self.gamma
        alpha = self.alpha

        with torch.no_grad():
            a_prime = self.policy_target.actor(s_prime_batch) #self.policy_target.get_action_and_log_prob(s_prime_batch, deterministic=False)
            target_Q = self.critic_target.Q(torch.cat((s_prime_batch, a_prime), dim=1))
            y = r_batch + gamma * target_Q
        
        current_Q = self.critic.Q(torch.cat((s_batch, a_batch), dim=1))

        # TODO: Implement Critic(s) update here
        critic_loss = F.mse_loss(current_Q, y)
        self.run_gradient_update_step(self.critic, critic_loss)

        # TODO: Implement Policy update here
        pred_action = self.policy.actor(s_batch)  #self.policy.get_action_and_log_prob(s_batch, deterministic=False)
        Q_a_policy = self.critic.Q(torch.cat((s_batch, pred_action), dim=1))
        policy_loss = -Q_a_policy.mean()
        self.run_gradient_update_step(self.policy, policy_loss)

        self.critic_target_update(base_net=self.critic.Q, target_net=self.critic_target.Q, tau=tau, soft_update=True)
        self.critic_target_update(base_net=self.policy.actor, target_net=self.policy_target.actor, tau=tau, soft_update=True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
