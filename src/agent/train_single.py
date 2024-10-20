import sys
import os
import torch

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import *
from src.environment import BackgammonEnv
from src.agent import BackgammonPPOAgent
from src.players import Player

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_agent(env, agent, num_episodes=NUM_EPISODES, max_timesteps=MAX_TIMESTEPS):
    batch_episodes = 10  # Number of episodes per batch
    batch_counter = 0

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        t = 0
        episode_reward = 0  # Initialize episode reward
        win = 0  # Initialize win flag (1 if Player1 wins, 0 otherwise)

        while not done and t < max_timesteps:
            t += 1

            # Get the current player
            current_player = env.current_player

            # Check if there are any legal actions
            action_mask = env.action_mask
            if action_mask.sum() == 0:
                # No legal actions, let the environment handle passing the turn
                # Take a null action; the environment will handle passing
                action = None
                observation, reward, done, info = env.step(action)
            else:
                # Select and perform an action
                action = agent.select_action(observation, action_mask)
                # Extract the scalar action from the array
                action = action[0]
                observation, reward, done, info = env.step(action)

                # Update the last experience in agent's memory with reward and done
                agent.memory[-1]["reward"] = reward
                agent.memory[-1]["done"] = done

            # Accumulate episode reward
            episode_reward += reward.item() if reward is not None else 0

            # Determine if PLAYER1 won
            if done and "winner" in info and "game_score" in info:
                winner = info["winner"]
                game_score = info["game_score"]
                win = 1 if winner == Player.PLAYER1 else 0  # 1 if PLAYER1 won, else 0

                # Optionally, categorize the game outcome
                if win:
                    if game_score == 1:
                        game_outcome = 0  # Win Normal
                    elif game_score == 2:
                        game_outcome = 1  # Win Gammon
                    elif game_score >= 3:
                        game_outcome = 2  # Win Backgammon
                else:
                    if game_score == 1:
                        game_outcome = 3  # Lose Normal
                    elif game_score == 2:
                        game_outcome = 4  # Lose Gammon
                    elif game_score >= 3:
                        game_outcome = 5  # Lose Backgammon

        # Add to the batch counter
        batch_counter += 1
        agent.total_episodes += 1  # For entropy annealing

        # Log metrics every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.log_metrics(episode + 1, episode_reward, win)

        # Update the agent after collecting enough episodes
        if batch_counter % batch_episodes == 0:
            agent.update()
            batch_counter = 0  # Reset batch counter
            # Optionally log progress to TensorBoard
            agent.writer.add_scalar("Loss/Policy Loss", agent.last_policy_loss, episode)
            agent.writer.add_scalar("Loss/Value Loss", agent.last_value_loss, episode)
            agent.writer.add_scalar("Loss/Entropy", agent.last_entropy_loss, episode)
            agent.writer.add_scalar("Loss/Total Loss", agent.last_total_loss, episode)
            agent.writer.add_scalar("Stats/Total Episodes", episode, episode)
            agent.writer.add_scalar("Stats/Total Steps", t, episode)
            agent.writer.add_scalar(
                "Stats/Entropy Coefficient", agent.entropy_coef, episode
            )
            print(f"Episode {episode}: Logged TensorBoard metrics.")

        # Entropy coefficient is updated within the agent's update method

        # Save model periodically
        if episode % 100_000 == 0 and episode > 0:
            filename = f"backgammon_ppo_episode_{episode}.pth"
            agent.save_model(filename=filename, to_s3=True)


if __name__ == "__main__":
    env = BackgammonEnv(device=device)
    agent = BackgammonPPOAgent(
        input_size=198,  # Adjust based on your observation space size
        hidden_size=HIDDEN_SIZE,
        action_size=env.action_space.n,
        entropy_coef_start=ENTROPY_COEF_START,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_anneal_episodes=ENTROPY_ANNEAL_EPISODES,
        device=device,
        s3_bucket_name=S3_BUCKET_NAME,
        s3_model_prefix=S3_MODEL_PREFIX,
        s3_log_prefix=S3_LOG_PREFIX,
    )

    # Optionally load a saved model
    agent.load_model(filename="backgammon_ppo_update_60.pth", from_s3=True)

    # Set agent to training mode
    agent.set_training_mode(training=True)

    # Start the training loop
    train_agent(env, agent, num_episodes=NUM_EPISODES, max_timesteps=MAX_TIMESTEPS)
