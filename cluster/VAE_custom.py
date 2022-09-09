import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from tqdm import tqdm

tfd = tfp.distributions

def load_data(fold=1, cheap=False):                                                                                                                                                                             
    features = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 5:]
    cheap_feat = features[:,:-8]
    cheap_feat[cheap_feat>0] = 1.
    cheap_feat[cheap_feat==0] = -1.
    exp_feat = features[:,-8:]
    cheap_feat = np.array(cheap_feat, np.float32) + np.random.normal(0, 0.01, (features.shape[0], 222)).astype(np.float32)
    top_20 = [127,  45, 193, 103, 116, 142, 138, 197,  92,   2,  93, 150, 129, 22,  60, 143, 115, 118,  15, 181]
    top_20.sort()
    cheap_feat = cheap_feat[:,top_20]
    exp_feat = np.array(features[:,-8:], np.float32).reshape((features.shape[0], 8))

    for i in range(exp_feat.shape[1]):
        exp_feat[:, i] = 2.*(exp_feat[:, i] - exp_feat[:, i].min()) / np.ptp(exp_feat[:, i]) - 1.
    if not cheap:                                                      
        data = np.concatenate((cheap_feat, exp_feat), axis=1)
    else:
        data = cheap_feat

    data = data.astype(np.float32)
    labels = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
    labels = labels.reshape(-1, 1)
    labels = labels.astype(np.float32)

    indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, fold]
    train_inputs = data[indices==0, :]   
    train_labels = labels[indices==0]   
    valid_inputs = data[indices==2, :]   
    valid_labels = labels[indices==2]   
    test_labels = labels[indices==1]  
    test_inputs = data[indices==1, :]  

    return (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)

def kl_divergence_nu(p_logits, q_logits): 
    cross_pq  = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(p_logits), logits=q_logits) 
    entropy_p = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(p_logits), logits=p_logits) 
    return cross_pq - entropy_p

def kl_divergence_z(mu0, Sigma0):
    """
    We assume mu1 and Sigma1 are 0 and I, respectively.
    """
    return 0.5 * (tf.math.reduce_sum(Sigma0, axis=-1) + tf.math.reduce_sum(mu0**2, axis=-1) - tf.cast(tf.shape(mu0)[-1], tf.float32) - tf.math.reduce_sum(tf.math.log(Sigma0+1e-8), axis=-1))

def elbo(mu0, Sigma0, mu1, Sigma1, decoder0, decoder1, q_logits, logits, cond0, cond1):
    """
    cond0: conditional distribution for the first decoder
    cond1: conditional distribution for the second decoder
    """
    elbo_z0 = decoder0 - kl_divergence_z(mu0, Sigma0)
    elbo_z1 = decoder1 - kl_divergence_z(mu1, Sigma1)
    new_logits = tf.repeat(tf.expand_dims(logits, axis=0), tf.shape(q_logits)[0], axis=0)
    q_probs = tf.nn.softmax(q_logits)
    return q_probs[:,0]*(elbo_z0 + cond0) + q_probs[:,1]*(elbo_z1 + cond1) - kl_divergence_nu(q_logits, new_logits)

class Decoder(tf.keras.Model):
    def __init__(self, size, nu, nhidden_layers=3, hidden_size=256):
        super(Decoder, self).__init__()

        self.size = size
        self.nu = nu

        self.nn_layers = [tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer='l2') for i in range(nhidden_layers)]

    def call(self, x, mu, Sigma):
        z_hat = tf.math.sqrt(Sigma) * tf.random.normal(tf.shape(mu)) + mu
        nu_tensor = tf.zeros((tf.shape(x)[0], 1)) + self.nu

        result = self.nn_layers[0](tf.concat([z_hat, nu_tensor], axis=1))
        for i in range(1, len(self.nn_layers)):
            result = self.nn_layers[i](result)

        mu_decoder = tf.math.abs(tf.keras.layers.Dense(self.size)(result))
        Sigma_decoder = tf.math.abs(tf.keras.layers.Dense(self.size)(result))

        prob1 = -0.5 * (self.size * tf.math.log(2*np.pi) + tf.math.reduce_sum(tf.math.log(Sigma_decoder+1e-8), axis=-1))
        prob2 = -0.5 * tf.math.reduce_sum((x - mu_decoder)**2 / (Sigma_decoder+1e-8), axis=-1)
        return tf.reshape(prob1 + prob2, (tf.shape(x)[0],))

class Conditional(tf.keras.Model):
    """
    The neural network finds mu and Sigma for the Gaussian distribution.
    """
    def __init__(self, nu, nhidden_layers=2, hidden_size=256):
        super(Conditional, self).__init__()

        self.nu = nu

        self.nn_layers = [tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer='l2') for i in range(nhidden_layers)]

    def call(self, x, y):
        nu_tensor = tf.zeros((tf.shape(x)[0], 1)) + self.nu

        result = self.nn_layers[0](tf.concat([x, nu_tensor], axis=1))
        for i in range(1, len(self.nn_layers)):
            result = self.nn_layers[i](result)

        mu = tf.math.abs(tf.keras.layers.Dense(1)(result))
        Sigma = tf.math.abs(tf.keras.layers.Dense(1)(result))

        prob = -0.5*tf.math.log(2*np.pi*Sigma**2) -0.5 * (y - mu)**2 / (Sigma**2 + 1e-8)
        return tf.reshape(prob, (tf.shape(prob)[0],))

class Encoder(tf.keras.Model):
    def __init__(self, size, nu, nhidden_layer=3, hidden_size=256):
        super(Encoder, self).__init__()

        self.size = size
        self.nu = nu

        self.nn_layers = [tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer='l2') for i in range(nhidden_layer)]

    def call(self, x):
        nu_tensor = tf.zeros((tf.shape(x)[0], 1)) + self.nu

        result = self.nn_layers[0](tf.concat([x, nu_tensor], axis=1))
        for i in range(1, len(self.nn_layers)):
            result = self.nn_layers[i](result)

        mu = tf.math.abs(tf.keras.layers.Dense(self.size)(result))
        Sigma = tf.math.abs(tf.keras.layers.Dense(self.size)(result))

        return mu, Sigma

def custom_activation(x):
    return tf.nn.tanh(x) * 0.5

class VAE(tf.keras.Model):
    def __init__(self, size, nhidden_layer=3, hidden_size=256):
        super(VAE, self).__init__()

        self.logits = tf.Variable(tf.constant([2.1972245, 0.0]), trainable=False, name='logits')

        self.encoder0 = Encoder(size, 0)
        self.encoder1 = Encoder(size, 1)
        self.decoder0 = Decoder(size, 0)
        self.decoder1 = Decoder(size, 1)
        self.conditional0 = Conditional(0)
        self.conditional1 = Conditional(1)

        layers = [tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer='l2') for i in range(nhidden_layer)] + [tf.keras.layers.Dense(2, activation=custom_activation)]
        self.q_net = tf.keras.Sequential(layers)

    def call(self, xcheap, xfull, y):
        mu0, Sigma0 = self.encoder0(xfull)
        mu1, Sigma1 = self.encoder1(xfull)
        decoder0 = self.decoder0(xfull, mu0, Sigma0)
        decoder1 = self.decoder1(xfull, mu1, Sigma1)
        cond0 = self.conditional0(xcheap, y)
        cond1 = self.conditional1(xfull, y)
        return elbo(mu0, Sigma0, mu1, Sigma1, decoder0, decoder1, self.q_net(tf.concat([xfull, y], axis=-1)), self.logits, cond0, cond1)

    def get_logits(self, x, y):
        return self.q_net(tf.concat([x, y], axis=-1))

def loss(model, cheap, full, y):
    return -tf.reduce_mean(model(cheap, full, y))

def grad(model, cheap, full, y):
    with tf.GradientTape() as tape: 
        loss_value = loss(model, cheap, full, y)
    return tape.gradient(loss_value, model.trainable_weights)

# If this is main file run the training
if __name__ == "__main__":
    # Load the data
    train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = load_data()
    cheap_train_inputs, cheap_train_labels, cheap_valid_inputs, cheap_valid_labels, cheap_test_inputs, cheap_test_labels = load_data(cheap=True)
    model = VAE(train_inputs.shape[1])
    model.load_weights('VAE_weights')

    lr = 1e-5

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    steps = 1000000
    loss_history = [] 
    for i in tqdm(range(steps)):
        grads = grad(model, cheap_train_inputs, train_inputs, train_labels)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if i % 100 == 0:
            print("Loss at step {:03d}: {:.5f}, Validation: {:.5f}".format(i, loss(model, cheap_train_inputs, train_inputs, train_labels), loss(model, cheap_valid_inputs, valid_inputs, valid_labels)))
        if i % 20000 and i > 0:
            tf.keras.backend.set_value(optimizer.learning_rate, max(optimizer.learning_rate * 0.75, 1e-5))

    model.save_weights('VAE_weights')

