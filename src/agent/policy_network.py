import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error


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
            Outputs a scalar logit for each input board state.

    Parameters:
        input_size (int): The size of the input feature vector.
                          Default is 198, corresponding to the board state representation.
        hidden_size (int): The number of neurons in the hidden fully connected layer.
                           Default is 128.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer mapping inputs to hidden representations.
        fc2 (nn.Linear): The second fully connected layer mapping hidden representations to output logits.

    Example:
        >>> model = BackgammonPolicyNetwork()
        >>> board_features = torch.randn(32, 198)  # Batch of 32 board states
        >>> logits = model(board_features)
        >>> probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    """

    def __init__(self, input_size=198, hidden_size=128):
        """
        Initializes the BackgammonPolicyNetwork.

        Args:
            input_size (int, optional): Size of the input feature vector. Default is 198.
            hidden_size (int, optional): Number of neurons in the hidden layer. Default is 128.
        """
        super(BackgammonPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Outputs a scalar value (logit)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor containing batch of board state features.
                              Shape: (batch_size, 198)

        Returns:
            torch.Tensor: Output tensor containing logits for each board state.
                          Shape: (batch_size,)

        """
        x = F.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = self.fc2(x)  # Output layer without activation
        return x.squeeze(-1)  # Remove last dimension if output shape is (N, 1)
