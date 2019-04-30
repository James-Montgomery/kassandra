import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None # remove stupid tf contrib warning
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_probability as tfp
tfd = tfp.distributions

# ---------------------------------------------------------------------------------------------------------------------
# Base Function
# ---------------------------------------------------------------------------------------------------------------------


def build(self, bayes):

    x = tf.placeholder(shape=(None, self._num_features), dtype=tf.float32, name="x")
    y = tf.placeholder(shape=(None, self._num_outputs), dtype=tf.float32, name="y")

    d = x
    for layer in range(self._layers):
        layer +=1

        l = tf.layers.dense(inputs=d,
                            units=self._units,
                            activation=self._activation,
                            kernel_regularizer=self._reg,
                            name="layer_{}".format(layer),
                            reuse=tf.AUTO_REUSE)

        d = tf.layers.dropout(inputs=l,
                              rate=self._dropout_rate,
                              noise_shape = (1, self._units),
                              training=bayes,
                              name="dropout_{}".format(layer))

        return x, y, d

# ---------------------------------------------------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------------------------------------------------


def build_mlp(self):

    x, y, d = build(self, False)

    mu = tf.layers.dense(inputs=d,
                         units=self._num_outputs,
                         activation=None,
                         name="mu",
                         reuse=tf.AUTO_REUSE)

    loss = tf.losses.mean_squared_error(y, mu)
    if self._reg is not None:
        loss += tf.losses.get_regularization_losses()

    return {
        "x" : x,
        "y" : y,
        "mu" : mu,
        "loss" : loss
    }

# ---------------------------------------------------------------------------------------------------------------------
# BNDropout
# ---------------------------------------------------------------------------------------------------------------------


def build_bn_dropout(self):

    if self._lengthscale is not None:
        #  TODO: Is this right for the heteroscedastic case?
        tau = 1.0
        N = self._num_train_samples
        reg = self._lengthscale**2 * (1 - self._dropout_rate) / (2. * N * tau)
        self._reg = tf.contrib.layers.l2_regularizer(reg)

    x, y, d = build(self, True)

    mu = tf.layers.dense(inputs=d,
                         units=self._num_outputs,
                         activation=None,
                         kernel_regularizer=self._reg,
                         name="mu",
                         reuse=tf.AUTO_REUSE)

    sigma = tf.layers.dense(inputs=d,
                            units=self._num_outputs,
                            activation=tf.nn.softplus,
                            kernel_regularizer=self._reg,
                            name="sigma",
                            reuse=tf.AUTO_REUSE)

    likelihood = tfd.Normal(loc=mu, scale=sigma)

    loss = tf.reduce_mean(-likelihood.log_prob(y))
    if self._reg is not None:
        loss += tf.losses.get_regularization_losses()

    return {
        "x" : x,
        "y" : y,
        "mu" : mu,
        "sigma" : sigma,
        "loss" : loss
    }

# ---------------------------------------------------------------------------------------------------------------------
# BNVI
# ---------------------------------------------------------------------------------------------------------------------


def VariationalParameter(name, shape, constraint=None, mean=0.0, std=1.0):
    means = tf.get_variable(name+'_mean', initializer = mean*tf.ones(shape), constraint=constraint)
    stds = tf.nn.softplus( tf.get_variable(name+'_std', initializer = std*tf.ones(shape)) )
    return tfd.Normal(loc=means, scale=stds)


def build_layer(self, input_tensor, w_shape, b_shape, layer_num, activation=None):

        # Prior
        pw = tfd.Normal(name='pw{}'.format(layer_num),
                        loc=self.p_mean*tf.ones(w_shape),
                        scale=self.p_std*tf.ones(w_shape))

        pb = tfd.Normal(name='pb{}'.format(layer_num),
                        loc=self.p_mean*tf.ones(b_shape),
                        scale=self.p_std*tf.ones(b_shape))

        # Variational Parameters
        qw = VariationalParameter('w{}'.format(layer_num), w_shape,
                                  mean=self.q_mean,
                                  std=self.q_std)

        qb = VariationalParameter('b{}'.format(layer_num), b_shape,
                                  mean=self.q_mean,
                                  std=self.q_std)

        # KL divergence with priors
        kl_w = tfp.distributions.kl_divergence(qw, pw,
                                               allow_nan_stats=True,
                                               name="kl_w{}".format(layer_num))

        kl_b = tfp.distributions.kl_divergence(qb, pb,
                                               allow_nan_stats=True,
                                               name="kl_b{}".format(layer_num))

        kl = tf.math.reduce_sum(kl_w) + tf.math.reduce_sum(kl_b)

        h =  tf.matmul(input_tensor, qw.sample()) + qb.sample()

        if activation is not None:
            h =  activation( h )

        return h, kl


def build_bn_vi(self):
    # Placeholders
    x = tf.placeholder(tf.float32, [None, self._num_features], name="x")
    y = tf.placeholder(tf.float32, [None, self._num_outputs], name="y")

    KL = []

    h, kl = build_layer(self, x, [self._num_features, self._units], [self._units], 0, self._activation)
    KL.append(kl)

    layer = -1
    for layer in range(self._layers):
        h, kl = build_layer(self, h, [self._units, self._units], [self._units], layer+1, self._activation)
        KL.append(kl)

    mu, kl = build_layer(self, h, [self._units, self._num_outputs], [self._num_outputs], layer+2)
    KL.append(kl)

    # Heteroskedastic
    sigma, kl = build_layer(self, h, [self._units, self._num_outputs], [self._num_outputs], "std", tf.nn.softplus)
    KL.append(kl)
    # Posterior Predictive
    pred_distribution = tfd.Normal(loc=mu, scale=sigma+1e-5)

    # Homoscedastic
    # sigma = VariationalParameter('std_', [self._num_outputs], mean=self.q_mean, std=self.q_std)
    # Posterior Predictive
    # pred_distribution = tfd.Normal(loc=mu, scale=tf.nn.softplus(sigma.sample()))

    # Probability of train data
    neg_log_prob = tf.reduce_mean(-pred_distribution.log_prob(y))

    prior_loss = 0.0
    for kl in KL:
        prior_loss += kl

    elbo_loss = neg_log_prob + prior_loss / self._num_train_samples

    return {
        'x' : x,
        'y' : y,
        'mu' : mu,
        'sigma' : sigma,
        'loss' : elbo_loss
    }
