
import tensorflow as tf
import numpy as np
from baselines.ssrl_pong.policies import CnnPolicy as Policy

class Model(object):

    def __init__(self, sess, ob_space, ac_space, batch_size=128, lr=5e-4, ent_coef=0.01, scope='parent'):
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.batch_size = batch_size
        self.scope = scope

        with tf.variable_scope(scope):
            self.act_model = Policy(sess=self.sess,
                                    ob_space=ob_space,
                                    ac_space=ac_space,
                                    nbatch=1,
                                    nsteps=1,
                                    reuse=tf.AUTO_REUSE)
            self.train_model = Policy(sess=self.sess,
                                      ob_space=ob_space,
                                      ac_space=ac_space,
                                      nbatch=batch_size,
                                      nsteps=1,
                                      reuse=tf.AUTO_REUSE)

            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
        self.ob_ph = self.train_model.X
        self.ac_ph = self.train_model.pdtype.sample_placeholder([None])
        self.ac = self.train_model.pd.sample()
        self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.pi, labels=self.ac_ph)

        self.loss = tf.reduce_mean(self.neglogpac) - ent_coef * tf.reduce_mean(self.train_model.entropy)
        #self.loss = tf.reduce_mean(self.neglogpac)

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))
        inc_step = self.global_step.assign_add(1)
        #self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5) 
        self._train = tf.group(self.trainer.apply_gradients(grads), inc_step)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def step(self, ob):
        a = self.act_model.step(ob)[0]
        return a

    def train(self, obs, acs):
        td_map = {self.ob_ph:obs, self.ac_ph:acs}
        train_loss, global_step, _ = self.sess.run([self.loss, self.global_step, self._train], td_map)
        #print("Training loss: {}".format(train_loss))
        return train_loss, global_step
        #print("Training loss: {}".format(train_loss))

    def copy(self, scope):
        #print('before:')
        #tvars = tf.trainable_variables()
        #tvars_vals = self.sess.run(tvars)

        #for var, val in zip(tvars, tvars_vals):
        #    if var.name.startswith(scope):
        #        print(var.name, val[0][0][0])
        #        break
        #for var, val in zip(tvars, tvars_vals):
        #    if var.name.startswith(self.scope):
        #        print(var.name, val[0][0][0])
        #        break

        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        self.sess.run(update_ops)      

        #print('After:')
        #tvars = tf.trainable_variables()
        #tvars_vals = self.sess.run(tvars)

        #for var, val in zip(tvars, tvars_vals):
        #    if var.name.startswith(scope):
        #        print(var.name, val[0][0][0])
        #        break
        #for var, val in zip(tvars, tvars_vals):
        #    if var.name.startswith(self.scope):
        #        print(var.name, val[0][0][0])
        #        break

