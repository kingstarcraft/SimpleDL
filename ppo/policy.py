import tensorflow as tf
import numpy as np


class PPO:
    def __init__(self, state_dim=3, action_dim=1,
                 action_lr=0.0001, critic_lr=0.0002,
                 action_step=10, critic_step=10):
        self._action_dim = action_dim
        self._state_dim = state_dim

        self._state = tf.placeholder(tf.float32, [None, self._state_dim], 'state')
        self._discount_reward = tf.placeholder(tf.float32, [None, 1], 'discount_rewrad')
        self._action = tf.placeholder(tf.float32, [None, self._action_dim], 'action')
        self._advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')

        self._lambda = tf.placeholder(tf.float32, None, 'lambda')

        self._kl_param = [0.01, 0.5]
        self._action_step = action_step
        self._critic_step = critic_step

        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self._state, 128, tf.nn.relu)
            self._value = tf.layers.dense(l1, 1)
            self._critic_advantage = self._discount_reward - self._value
            self._critic_loss = tf.reduce_mean(tf.square(self._critic_advantage))
            self._critic_train = tf.train.AdamOptimizer(critic_lr).minimize(self._critic_loss)

        with tf.variable_scope("actor"):
            self._input_state, self._state_feature = self._create_state_feature(self._state_dim)
            self._output = self._create_actors(self._state_feature, self._action_dims)

        policy, policy_param = self._bulid_angent('policy', trainable=True)
        old_policy, old_policy_param = self._bulid_angent('old_policy', trainable=False)

        with tf.variable_scope('sample_action'):
            self._sample_op = tf.squeeze(policy.sample(1), axis=0)
        with tf.variable_scope('update_old_policy'):
            self._update_old_policy = [old_ply.assign(ply) for ply, old_ply in zip(policy_param, old_policy_param)]

        with tf.variable_scope('action'):
            ratio = policy.prob(self._action) / old_policy.prob(self._action)
            surrogate = ratio * self._advantage
            kl = tf.distributions.kl_divergence(old_policy, policy)
            self._kl_mean = tf.reduce_mean(kl)
            self._action_loss = -tf.reduce_mean(surrogate - self._lambda * kl)
            self._action_train = -tf.train.AdamOptimizer(action_lr).minimize(self._action_loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def _update(self, state, action, reward):
        self._sess.run(self._update_old_policy)
        advantage = self._sess.run(self._critic_advantage, {self._state: state, self._discount_reward: reward})
        for _ in range(self._action_step):
            _, loss, kl = self._sess.run([self._action_train, self._action_loss, self._kl_mean],
                                         {self._state: state, self._action: action,
                                          self._advantage: advantage, self._lambda: self._kl_param[1]})
            if kl > 4 * self._kl_param[0]:
                break
            elif kl < self._kl_param[0] / 1.5:
                self._kl_param[1] /= 2
            elif kl > self._kl_param[1]:
                self._kl_param[1] *= 2
            self._kl_param[1] = np.clip(self._kl_param[1], 1e-10, 10)
            print('action loss = %f, kl = %f.' % loss, kl)
        print('------------------------------------------')
        for _ in range(self._critic_step):
            _, loss, advantage = self._sess.run([self._critic_train, self._critic_advantage, self._critic_loss],
                                                {self._state: state, self._discount_reward: reward})
        print('==========================================')

    def _bulid_angent(self, name, trainable):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(self._state, 128, tf.nn.relu, trainable=trainable)
            mean = 2 * tf.layers.dense(layer1, self._action_dim, tf.nn.tanh, trainable=trainable)
            variance = tf.layers.dense(layer1, self._action_dim, tf.nn.softplus, trainable=trainable)
            policy = tf.distributions.Normal(loc=mean, scale=variance)
        return policy, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
