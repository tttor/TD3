import torch

class ActionValueNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ActionValueNetwork, self).__init__()
        self.type = 'actionvalue'
        output_dim = 1 # one output neuron estimating the value of any action
        self.hidden_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
            torch.nn.ReLU()
        )
        self.output_net = torch.nn.Linear(hidden_dim[1], output_dim)

    def forward(self, observ, action):
        action_value = self.output_net( self.hidden_net(torch.cat([observ, action], dim=1)) )
        return action_value

class DeterministicPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeterministicPolicyNetwork, self).__init__()
        self.type = 'policy'
        self.hidden_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
            torch.nn.Tanh()
        )
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim[1], output_dim),
            torch.nn.Tanh()
        )

    def forward(self, observ):
        action = self.output_net(self.hidden_net(observ)) # is a deterministic action
        return action
