"""
Gymnasium Environment Runner for DQN and DDQN
This script trains and tests DQN/DDQN agents on Gymnasium environments
with Weights & Biases tracking and video recording.
"""

import gymnasium as gym
import torch
import numpy as np
import wandb
import argparse
import os
from datetime import datetime
from gymnasium.wrappers import RecordVideo

from models import DQNAgent, DDQNAgent


def generate_run_name(algorithm, env_name, learning_rate, gamma, epsilon_decay, buffer_size, batch_size):
    """
    Generate a descriptive name for the run that includes key hyperparameters.
    
    Args:
        algorithm: DQN or DDQN
        env_name: Environment name
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_decay: Epsilon decay rate
        buffer_size: Replay buffer size
        batch_size: Batch size
    
    Returns:
        Formatted run name string
    """
    env_short = env_name.split('-')[0]  # e.g., CartPole from CartPole-v1
    return (f"{algorithm}_{env_short}_"
            f"lr{learning_rate}_g{gamma}_"
            f"ed{epsilon_decay}_buf{buffer_size//1000}k_bs{batch_size}")


def create_environment(env_name, render_mode=None, video_folder=None, episode_trigger=None, name_prefix=None):
    """
    Create a Gymnasium environment with optional video recording.
    
    Args:
        env_name: Name of the environment
        render_mode: Render mode (None, 'rgb_array', 'human')
        video_folder: Folder to save videos (None for no recording)
        episode_trigger: Function to determine which episodes to record
        name_prefix: Prefix for video file names
    
    Returns:
        Gymnasium environment
    """
    env = gym.make(env_name, render_mode=render_mode if video_folder else None)
    
    if video_folder:
        if name_prefix is None:
            name_prefix = f"{env_name}-rl-video"
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=episode_trigger if episode_trigger else lambda x: True,
            name_prefix=name_prefix
        )
    
    return env


def get_action_space_info(env):
    """Get information about the action space."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n, True
    else:
        # For continuous action spaces, discretize
        return 10, False


def discretize_action(action, env, num_bins=10):
    """Discretize continuous actions."""
    if isinstance(env.action_space, gym.spaces.Box):
        low = env.action_space.low[0]
        high = env.action_space.high[0]
        bin_size = (high - low) / num_bins
        continuous_action = low + (action + 0.5) * bin_size
        return np.array([continuous_action])
    return action


def train_agent(
    agent,
    env_name,
    run_name,
    num_episodes=1000,
    max_steps=500,
    log_interval=10,
    save_dir='models',
    use_wandb=True
):
    """
    Train the agent on the specified environment.
    
    Args:
        agent: DQN or DDQN agent
        env_name: Name of the Gymnasium environment
        run_name: Name for this run (used for saving models)
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        log_interval: Interval for logging to W&B
        save_dir: Directory to save models
        use_wandb: Whether to use Weights & Biases logging
    
    Returns:
        Trained agent
    """
    env = create_environment(env_name)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Discretize action if needed
            env_action = discretize_action(action, env)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_loss = np.mean(episode_losses[-log_interval:])
            current_epsilon = agent.get_epsilon()
            
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {current_epsilon:.4f}")
            
            if use_wandb:
                wandb.log({
                    'episode': episode + 1,
                    'avg_reward': avg_reward,
                    'episode_reward': episode_reward,
                    'avg_loss': avg_loss,
                    'epsilon': current_epsilon
                })
    
    # Save final model with descriptive name
    model_path = os.path.join(save_dir, f"{run_name}.pt")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()
    return agent


def test_agent(
    agent,
    env_name,
    run_name,
    num_tests=100,
    max_steps=500,
    render=False,
    record_video=False,
    video_folder='videos'
):
    """
    Test the trained agent.
    
    Args:
        agent: Trained DQN or DDQN agent
        env_name: Name of the Gymnasium environment
        run_name: Name for this run (used for video naming)
        num_tests: Number of test episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        record_video: Whether to record videos
        video_folder: Folder to save videos
    
    Returns:
        Dictionary with test results
    """
    # Create environment
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = create_environment(
            env_name,
            render_mode='rgb_array',
            video_folder=video_folder,
            episode_trigger=lambda x: x % 10 == 0,  # Record every 10th episode
            name_prefix=run_name
        )
    else:
        env = create_environment(env_name, render_mode='human' if render else None)
    
    test_rewards = []
    test_durations = []
    
    for test_episode in range(num_tests):
        state, _ = env.reset()
        episode_reward = 0
        episode_duration = 0
        
        for step in range(max_steps):
            # Select action (greedy, no exploration)
            action = agent.select_action(state, epsilon=0.0)
            
            # Discretize action if needed
            env_action = discretize_action(action, env)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_duration += 1
            state = next_state
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_durations.append(episode_duration)
        
        if (test_episode + 1) % 10 == 0:
            print(f"Test Episode {test_episode + 1}/{num_tests} - "
                  f"Reward: {episode_reward:.2f}, Duration: {episode_duration}")
    
    env.close()
    
    results = {
        'mean_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'mean_duration': np.mean(test_durations),
        'std_duration': np.std(test_durations),
        'min_reward': np.min(test_rewards),
        'max_reward': np.max(test_rewards)
    }
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Duration: {results['mean_duration']:.2f} ± {results['std_duration']:.2f}")
    print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("="*50 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train DQN/DDQN on Gymnasium environments')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1'],
                        help='Gymnasium environment')
    parser.add_argument('--algorithm', type=str, default='DQN',
                        choices=['DQN', 'DDQN'],
                        help='Algorithm to use')
    
    # Training settings
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of test episodes')
    
    # Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=1000,
                        help='Epsilon decay constant (higher = slower decay)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--target-update-freq', type=int, default=10,
                        help='Target network update frequency')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer sizes')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test, do not train')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to model to load')
    parser.add_argument('--record-video', action='store_true',
                        help='Record videos during testing')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment to get dimensions
    temp_env = gym.make(args.env)
    state_size = temp_env.observation_space.shape[0]
    action_size, is_discrete = get_action_space_info(temp_env)
    temp_env.close()
    
    # Initialize W&B before constructing the agent so sweep configs can override args
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project='rl-dqn-gymnasium',
            config=vars(args)
        )
        if wandb_run is not None:
            # Update argparse namespace with sweep-provided values
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)

    # Display final configuration after potential sweep overrides
    print(f"\nEnvironment: {args.env}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Action space: {'Discrete' if is_discrete else 'Continuous (discretized)'}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Device: {args.device}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})\n")

    # Create agent after potential sweep overrides have been applied
    AgentClass = DQNAgent if args.algorithm == 'DQN' else DDQNAgent
    agent = AgentClass(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        hidden_sizes=args.hidden_sizes,
        device=args.device
    )

    # Generate descriptive run name with key hyperparameters
    run_name = generate_run_name(
        algorithm=args.algorithm,
        env_name=args.env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
    )

    if wandb_run is not None:
        wandb_run.name = run_name
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)
    
    # Train or test
    if not args.test_only:
        print("Starting training...")
        agent = train_agent(
            agent=agent,
            env_name=args.env,
            run_name=run_name,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            log_interval=10,
            save_dir='models',
            use_wandb=use_wandb
        )
    
    # Test
    print("\nStarting testing...")
    test_results = test_agent(
        agent=agent,
        env_name=args.env,
        run_name=run_name,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        render=args.render,
        record_video=args.record_video,
        video_folder=f'videos'
    )
    
    # Log test results to W&B
    if use_wandb:
        wandb.log({
            'test_mean_reward': test_results['mean_reward'],
            'test_std_reward': test_results['std_reward'],
            'test_mean_duration': test_results['mean_duration'],
            'test_std_duration': test_results['std_duration']
        })
        wandb.finish()
    
    print("\nTraining and testing complete!")


if __name__ == '__main__':
    main()
