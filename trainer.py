if __name__ == "__main__":
    scores = train_dqn_agent(
        env_name="Breakout-v4", 
        num_episodes=2000,  # Adjust as needed
        max_steps=10000,
        render=False
    )
