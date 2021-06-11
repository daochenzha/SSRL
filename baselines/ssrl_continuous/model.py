
import tensorflow as tf
import numpy as np
from baselines.ssrl_continuous.mlp_policy import MlpPolicy as Policy
from baselines.common.mpi_adam import MpiAdam
from baselines.common import set_global_seeds, tf_util as U

class Model(object):

    def __init__(self, ob_space, ac_space, lr=5e-4, ent_coef=0.00):
        self.sess = tf.get_default_session()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.lr = lr

        self.pi = Policy(name="pi",
                         ob_space=ob_space,
                         ac_space=ac_space,
                         reuse=False,
                         hid_size=64,
                         num_hid_layers=2)
        ob = U.get_placeholder_cached(name="ob")
        ac = self.pi.pdtype.sample_placeholder([None])
        stochastic = U.get_placeholder_cached(name="stochastic")
        loss = tf.reduce_mean(tf.square(ac-self.pi.ac))
        var_list = self.pi.get_trainable_variables()
        self.adam = MpiAdam(var_list)
        self.lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

        self.loss = tf.reduce_mean(tf.square(ac-self.pi.ac)) - ent_coef * tf.reduce_mean(self.pi.pd.entropy())

    def step(self, ob):
        return self.pi.act(True, ob)[0]

    def train(self, obs, acs):
        train_loss, g = self.lossandgrad(obs, acs, True)
        self.adam.update(g, self.lr)
        return train_loss
        #print("Training loss: {}".format(train_loss))


