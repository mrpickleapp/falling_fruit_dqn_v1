import numpy as np
import tensorflow as tf
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, epsilon=1e-6, alpha=0.2, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha   # how much prioritisation is used
        self.beta = beta    # for importance sampling weights
        self.beta_increment = beta_increment
        self.priority_buffer = np.zeros(self.capacity)
        self.data = []
        self.position = 0

    def length(self):
        return len(self.data)

    def push(self, experience):
        max_priority = np.max(self.priority_buffer) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            self.data[self.position] = experience
        self.priority_buffer[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priority_buffer[:len(self.data)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.data), batch_size, p=probabilities)
        experiences = [self.data[i] for i in indices]

        total = len(self.data)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1., self.beta + self.beta_increment])
        
        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priority_buffer[idx] = error + self.epsilon

class DQN:
    def __init__(self, state_shape, action_size, learning_rate_max=0.001, learning_rate_decay=0.995, gamma=0.75, 
                 memory_size=2000, batch_size=32, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.state_shape = state_shape
        self.state_tensor_shape = (-1,) + state_shape
        self.action_size = action_size
        self.learning_rate_max = learning_rate_max
        self.learning_rate = learning_rate_max
        self.learning_rate_decay = learning_rate_decay
        self.gamma = gamma
        self.memory_size = memory_size
        # self.memory = deque(maxlen=memory_size)
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=self.state_shape))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear', name='action_values', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state, epsilon=None):
        if epsilon == None:
            epsilon = self.exploration_rate
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.target_model.predict(state, verbose=0)[0])
    
    def replay(self, episode=0):

        if self.memory.length() < self.batch_size:
            return None
        
        experiences, indices, weights = self.memory.sample(self.batch_size)
        unpacked_experiences = list(zip(*experiences))
        states, actions, rewards, next_states, dones = [list(arr) for arr in unpacked_experiences]

        # Convert to tensors
        states = tf.convert_to_tensor(states)
        states = tf.reshape(states, self.state_tensor_shape)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        next_states = tf.reshape(next_states, self.state_tensor_shape)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute Q values and next Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        q_values = self.model.predict(states, verbose=0)

        # Compute target values using the Bellman equation
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_target_q_values

        # Compute TD errors
        batch_indices = np.arange(self.batch_size)
        q_values_current_action = q_values[batch_indices, actions]
        td_errors = targets - q_values_current_action
        self.memory.update_priorities(indices, np.abs(td_errors))

        # For learning: Adjust Q values of taken actions to match the computed targets
        q_values[batch_indices, actions] = targets

        loss = self.model.train_on_batch(states, q_values, sample_weight=weights)

        self.exploration_rate = self.exploration_max*self.exploration_decay**episode
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
        self.learning_rate = self.learning_rate_max*self.learning_rate_decay**episode
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate)

        return loss
    
    def load(self, name):
        self.model = tf.keras.models.load_model(name)
        self.target_model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)