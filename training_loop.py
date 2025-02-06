def stack_frames(frames, new_frame, is_new_episode=False):
    """
    Stack frames for the DQN input. Typically we keep the last 4 frames.
    'frames' is a deque with maxlen=4. 
    """
    if is_new_episode:
        # Clear the deque and stack the same frame 4 times
        frames.clear()
        for _ in range(4):
            frames.append(new_frame)
    else:
        frames.append(new_frame)
    
    # Return stacked frames as a numpy array of shape (4, 84, 84)
    return np.stack(frames, axis=0)

def train_dqn_agent(
    env_name="Breakout-v4", 
    num_episodes=5000,
    max_steps=10000,
    render=False
):
    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)
    # Possible alternative: env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    
    num_actions = env.action_space.n
    agent = DQNAgent(num_actions=num_actions)
    
    frame_stack = deque(maxlen=4)
    
    best_score = -float("inf")
    scores = []
    
    for episode in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        # Preprocess
        processed_state = preprocess_frame(state)
        # Stack frames
        stacked_state = stack_frames(frame_stack, processed_state, is_new_episode=True)
        
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            action = agent.select_action(stacked_state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Preprocess
            processed_next_state = preprocess_frame(next_state)
            stacked_next_state = stack_frames(frame_stack, processed_next_state, is_new_episode=False)
            
            agent.store_transition(stacked_state, action, reward, stacked_next_state, done)
            agent.learn()
            agent.update_epsilon()
            
            stacked_state = stacked_next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        scores.append(episode_reward)
        
        # Track best score
        if episode_reward > best_score:
            best_score = episode_reward
        
        print(f"Episode {episode} | Score: {episode_reward} | Best Score: {best_score} | Epsilon: {agent.epsilon:.4f}")
        
        # (Optional) Save model
        # if (episode + 1) % 100 == 0:
        #     torch.save(agent.online_network.state_dict(), f"dqn_breakout_{episode+1}.pth")
        
    env.close()
    return scores
