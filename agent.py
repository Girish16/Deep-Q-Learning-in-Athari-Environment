class DQNAgent:
    def __init__(
        self, 
        num_actions,
        gamma=0.99,
        lr=1e-4,
        batch_size=32,
        buffer_size=100000,
        min_epsilon=0.1,
        epsilon_decay=1e-6,
        update_target_freq=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq
        self.device = device
        
        # Networks
        self.online_network = DQN(num_actions).to(self.device)
        self.target_network = DQN(num_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Internal step counter
        self.learn_step_counter = 0
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        'state' is expected to be a stack of 4 frames with shape (4, 84, 84).
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            # Convert state to a batch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
    
    def learn(self):
        # Only learn if we have enough samples
        if self.replay_buffer.size() < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q estimates
        q_values = self.online_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Next Q values (Double DQN approach):
        # 1. Get action that maximizes Q from online network
        online_next_q = self.online_network(next_states_t)
        max_actions = torch.argmax(online_next_q, dim=1)
        
        # 2. Evaluate those actions using the target network
        target_next_q = self.target_network(next_states_t)
        selected_q_values = target_next_q.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        
        # If not using Double DQN, you could just do: target_next_q = self.target_network(next_states_t).max(1)[0]
        
        # Calculate target
        target = rewards_t + (1 - dones_t) * self.gamma * selected_q_values
        
        # Loss
        loss = F.mse_loss(q_values, target.detach())
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
