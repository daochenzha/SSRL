
import tensorflow as tf
import numpy as np
from baselines.ssrl_discrete.policies import MlpPolicy as Policy

class Model(object):

    def __init__(self, ob_space, ac_space, batch_size=128, lr=5e-4, ent_coef=0.0):
        self.sess = tf.get_default_session()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.batch_size = batch_size

        self.act_model = Policy(sess=self.sess,
                                ob_space=ob_space,
                                ac_space=ac_space,
                                nbatch=1,
                                nsteps=1,
                                reuse=False)
        self.train_model = Policy(sess=self.sess,
                                  ob_space=ob_space,
                                  ac_space=ac_space,
                                  nbatch=batch_size,
                                  nsteps=1,
                                  reuse=True)

        self.ob_ph = self.train_model.X
        self.ac_ph = self.train_model.pdtype.sample_placeholder([None])
        self.ac = self.train_model.pd.sample()
        self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.pi, labels=self.ac_ph)

        #self.loss = tf.reduce_mean(self.neglogpac) - ent_coef * tf.reduce_mean(self.train_model.entropy)
        self.loss = tf.reduce_mean(self.neglogpac)

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads = list(zip(grads, params))
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self._train = self.trainer.apply_gradients(grads)

        tf.global_variables_initializer().run(session=self.sess)

    def step(self, ob):
        a = self.act_model.step(np.expand_dims(ob, axis=0))[0][0]
        return a

    def train(self, obs, acs):
        #print(obs.shape, acs.shape)
        td_map = {self.ob_ph:obs, self.ac_ph:acs}
        train_loss, _ = self.sess.run([self.loss, self._train], td_map)
        #print("Training loss: {}".format(train_loss))
        return train_loss
        #print("Training loss: {}".format(train_loss))


