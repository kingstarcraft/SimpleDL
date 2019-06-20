import tensorflow as tf
import numpy as np


def build_policy(state, action_dim, trainable=False):
    layer1 = tf.layers.dense(state, 100, tf.nn.relu, trainable=trainable)
    mean = 2 * tf.layers.dense(layer1, action_dim, tf.nn.tanh, trainable=trainable)
    variance = tf.layers.dense(layer1, action_dim, tf.nn.softplus, trainable=trainable)
    policy = tf.distributions.Normal(loc=mean, scale=variance)
    return policy


def build_cirtic(state, reward):
    l1 = tf.layers.dense(state, 100, tf.nn.relu)
    value = tf.layers.dense(l1, 1)
    advantage = reward - value
    return advantage, value




class PPO:
    def __init__(self, state_dim, action_dim,
                 action_step = 10, circle_step = 10,
                 action_lr=0.0001, critic_lr=0.0002, epsilon=0.2):

        #    self._action_dim = action_dim
        #     self._state_dim = state_dim

        self._state = tf.placeholder(tf.float32, [None, state_dim], 'state')
        self._reward = tf.placeholder(tf.float32, [None, 1], 'discount_rewrad')
        self._action = tf.placeholder(tf.float32, [None, action_dim], 'action')

        self._circle_step = circle_step
        self._action_step = action_step

     #   self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        #   self._advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # self._lambda = tf.placeholder(tf.float32, None, 'lambda')
        # self._kl_param = [0.01, 0.5]
        with tf.variable_scope('cirtic'):
            self._advantage, self._value = build_cirtic(self._state, self._reward)
            self._critic_loss = tf.reduce_mean(tf.square(self._advantage))
            self._train_critic = tf.train.AdamOptimizer(critic_lr).minimize(self._critic_loss)

        with tf.variable_scope('action'):
            with tf.variable_scope('policy'):
                policy = build_policy(self._state, action_dim,trainable=True)
            with tf.variable_scope('old_policy'):
                old_policy = build_policy(self._state, action_dim, trainable=False)
            policy_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action/policy')
            old_policy_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action/old_policy')


            ratio = policy.prob(self._action) / old_policy.prob(self._action)
            surrogate = ratio * self._advantage

            self._action_loss = -tf.reduce_mean(tf.minimum(
                surrogate, tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self._advantage))
            self._train_action = tf.train.AdamOptimizer(action_lr).minimize(self._action_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='action'))

        with tf.variable_scope('action_update'):
            self._update_action = [old_ply.assign(ply) for ply, old_ply in zip(policy_param, old_policy_param)]

        with tf.variable_scope('sample_action'):
            self._sample = tf.squeeze(policy.sample(1), axis=0)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, state, action, reward):
        #glob_step = self._sess.run(self._glob_step)
        self._sess.run(self._update_action)

        for time in range(self._action_step):
            value, action_loss, _ = self._sess.run((
                self._value, self._action_loss, self._train_action),
                {self._action:action, self._state:state, self._reward: reward})
        for time in range(self._circle_step):
            value, critic_loss, _ = self._sess.run((
                self._value, self._critic_loss, self._train_critic),
                {self._state:state, self._reward: reward})
        #self._sess.run(self._glob_step.assign(glob_step + 1))
        #return glob_step

    def action(self, state):
        sample = self._sess.run(self._sample, {self._state: [state]})[0]
        return np.clip(sample, -2, 2)

    def value(self, state):
        value = self._sess.run(self._value, {self._state: [state]})
        return value[0,0]

if __name__ == "__main__":
    max_epoch = 1000
    iter_size = 200
    batch_size = 32

    ppo = PPO(3, 1)
    import gym
    env = gym.make('Pendulum-v0').unwrapped
    epoch_rewards = []

    for epoch in range(max_epoch):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        epoch_reward = 0
        for iter in range(iter_size):
            env.render()
            states.append(state)
            action = ppo.action(state)
            state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append((reward+8) / 8)
            epoch_reward += reward

            if (iter +1)%batch_size == 0 or iter == iter_size - 1:
                value = ppo.value(state)
                discounted_rewards = []
                for r in rewards[::-1]:
                    value = r + 0.9*value
                    discounted_rewards.append(value)
                discounted_rewards.reverse()
                ppo.train(np.vstack(states), np.vstack(actions), np.array(discounted_rewards).reshape([-1, 1]))
                states = []
                actions = []
                rewards = []
        epoch_rewards.append(epoch_reward)
        print('epoch %04d: %05d.' % (epoch, epoch_reward))

    from matplotlib import pyplot as plt
    plt.plot(epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Rewards')
    plt.show()
