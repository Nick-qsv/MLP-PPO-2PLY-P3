import torch  # pylint: disable=import-error
from typing import Tuple


def get_all_dice_rolls_tensor() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates all unique sorted dice rolls with their probabilities as PyTorch tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - dice_rolls_tensor: Tensor of shape (21, 2) containing all unique dice roll combinations.
            - probabilities_tensor: Tensor of shape (21,) containing the probability of each dice roll.
    """
    dice_rolls = []
    counts = []
    total_outcomes = 36  # Total possible outcomes with two dice

    # Generate all unique sorted dice rolls and count their occurrences
    for die1 in range(1, 7):
        for die2 in range(die1, 7):  # Ensure die2 >= die1 to avoid duplicates
            dice_rolls.append([die1, die2])
            if die1 == die2:
                counts.append(1)  # Doubles occur once
            else:
                counts.append(2)  # Non-doubles occur twice

    # Convert lists to PyTorch tensors
    dice_rolls_tensor = torch.tensor(dice_rolls, dtype=torch.int32)  # Shape: (21, 2)
    counts_tensor = torch.tensor(counts, dtype=torch.float32)  # Shape: (21,)

    # Calculate probabilities
    probabilities_tensor = counts_tensor / total_outcomes  # Shape: (21,)

    return dice_rolls_tensor, probabilities_tensor
