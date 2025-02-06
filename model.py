class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # 4 input frames
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the shape of the flattened feature after conv layers
        # If input is (4, 84, 84), we can do a quick forward pass or reason about it:
        # conv1 -> out shape [32, 20, 20]
        # conv2 -> out shape [64, 9, 9]
        # conv3 -> out shape [64, 7, 7]
        # flattened size = 64*7*7 = 3136
        self.fc_input_dim = 64 * 7 * 7
        
        # Dueling network splits into two streams:
        # Value stream
        self.value_fc = nn.Linear(self.fc_input_dim, 512)
        self.value_out = nn.Linear(512, 1)
        # Advantage stream
        self.advantage_fc = nn.Linear(self.fc_input_dim, 512)
        self.advantage_out = nn.Linear(512, num_actions)
        
    def forward(self, x):
        """
        x shape: (batch_size, 4, 84, 84)
        """
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(-1, self.fc_input_dim)
        
        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)
        
        # Advantage stream
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a)
        
        # Combine streams: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        
        return q
