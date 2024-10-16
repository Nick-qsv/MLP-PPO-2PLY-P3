import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgammonPolicyNetwork(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) Policy Network for Backgammon.

    This network evaluates backgammon board states and outputs logits
    that can be transformed into a probability distribution over possible
    actions/moves. It is designed to be integrated with a Proximal Policy
    Optimization (PPO) agent and supports batch processing for efficiency.

    Architecture:
        - Input Layer:
            Accepts board states represented as tensors of shape (N, 198),
            where N is the batch size.
        - Hidden Layer:
            A single fully connected layer with a specified number of neurons
            and ReLU activation.
        - Output Layer:
            Outputs logits for each possible action.

    Parameters:
        input_size (int): The size of the input feature vector.
                          Default is 198, corresponding to the board state representation.
        hidden_size (int): The number of neurons in the hidden fully connected layer.
                           Default is 128.
        action_size (int): The number of possible actions.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer mapping inputs to hidden representations.
        action_head (nn.Linear): The fully connected layer mapping hidden representations to action logits.
        value_head (nn.Linear): The fully connected layer mapping hidden representations to state value estimates.

    Example:
        >>> model = BackgammonPolicyNetwork(input_size=198, hidden_size=128, action_size=10)
        >>> board_features = torch.randn(32, 198)  # Batch of 32 board states
        >>> logits, state_values = model(board_features)
        >>> action_probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    """

    def __init__(self, input_size=198, hidden_size=128, action_size=500):
        """
        Initializes the BackgammonPolicyNetwork.

        Args:
            input_size (int, optional): Size of the input feature vector. Default is 198.
            hidden_size (int, optional): Number of neurons in the hidden layer. Default is 128.
            action_size (int): Number of possible actions.
        """
        super(BackgammonPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor containing batch of board state features.
                              Shape: (batch_size, 198)

        Returns:
            logits (torch.Tensor): Tensor containing logits for each possible action.
                                   Shape: (batch_size, action_size)
            state_values (torch.Tensor): Tensor containing state value estimates.
                                   Shape: (batch_size,)
        """
        x = F.relu(self.fc1(x))  # Hidden layer with ReLU activation
        logits = self.action_head(x)  # Action logits
        state_values = self.value_head(x).squeeze(-1)  # State value estimates
        return logits, state_values
