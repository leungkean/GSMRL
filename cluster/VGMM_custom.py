import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from tqdm import tqdm

tfd = tfp.distributions

def load_data(fold=1, cheap_only=False):                                                                                                                                                                             
    features = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 5:]
    cheap_feat = features[:,:-8]
    cheap_feat[cheap_feat>0] = 1.
    cheap_feat[cheap_feat==0] = -1.
    exp_feat = features[:,-8:]
    cheap_feat = np.array(cheap_feat, np.float32) + np.random.normal(0, 0.01, (features.shape[0], 222)).astype(np.float32)
    exp_feat = np.array(features[:,-8:], np.float32).reshape((features.shape[0], 8))
    # Normalize expensive feature to [-1, 1]
    for i in range(exp_feat.shape[1]):
        exp_feat[:, i] = 2.*(exp_feat[:, i] - exp_feat[:, i].min()) / np.ptp(exp_feat[:, i]) - 1.
    if not cheap_only:                                                      
        data = np.concatenate((cheap_feat, exp_feat), axis=1)
    else:
        data = cheap_feat
 
    labels = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
    labels = labels.reshape(-1, 1)
 
    indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, fold]
    train_inputs = data[indices==0, :]   
    train_labels = labels[indices==0]   
    valid_inputs = data[indices==2, :]   
    valid_labels = labels[indices==2]   
    test_labels = labels[indices==1]  
    test_inputs = data[indices==1, :]  
    return (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)

def kl_divergence(p_logits, q_logits):
    """ Compute the KL divergence between discrete distributions p and q.
    Args:
        p_logits: N x m tensor of logits for N distributions over m elements.
        q_logits: N x m  tensor of logits for N distributions over m elements.
    Return:
        kl: N tensor of KL(p_i || q_i)
    """
    """ Hint: it is numerically stable to use
    tf.nn.softmax_cross_entropy_with_logits. Be careful about what is in logit
    versus probability space.
    """
    cross_pq  = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(p_logits), logits=q_logits)
    entropy_p = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(p_logits), logits=p_logits)
    return cross_pq - entropy_p  # TODO

def decoder_z(x, k, nhidden_layers=2, hidden_size=100):
    """ Compute the decoder using a neural network.
    Args:
        x: N x d tensor input.
    Return:
        log_prob: N x k tensor of log probabilities.
    """
    layers = [tf.keras.layers.Dense(hidden_size, activation='elu') for i in range(nhidden_layers)] + [tf.keras.layers.Dense(k, activation='sigmoid')]
    decode_net = tf.keras.Sequential(layers)
    return decode_net(x)

def elbo_z(inputs, q_logits, means, sigmas):
    """ Compute the ELBO for a given GMM model and approximate posterior.
    Args:
        inputs: N x d points to compute ELBO at.
        q_logits: N x ncomp tensor of logits for approximate posterior of z_i.
        means: ncomp x 1 x d tensor of means of mixture model.
        sigmas: ncomp x 1 tensor of std. dev. of mixture model.
    Return:
        kl: N tensor of KL(p_i || q_i)
    """
    """
    Here logits are set to be N(0,1)
    """
    norm_q = tf.nn.softmax(q_logits)
    decode = decoder(inputs, means, sigmas)
    new_logits = tf.repeat(tf.expand_dims(logits,axis=0), repeats=[inputs.shape[0]], axis=0)
    return tf.math.reduce_sum(norm_q*decode, axis=1) + kl_divergence(q_logits, new_logits)  # TODO

class VGMM_z(tf.keras.Model):
    def __init__(self, d, k=100, nhidden_layers=3, hidden_size=256):
        super(VGMM_z, self).__init__()
        """
        Setup a neural network with nhidden_layers with hidden_size number of units
        to output logits of approximate posteriors below.
        Hint: use tf.keras.layers.Dense and a ELU activation where appropriate.
        layers = [...]
        """
        layers = [tf.keras.layers.Dense(hidden_size, activation='elu') for i in range(nhidden_layers)] + [tf.keras.layers.Dense(k)]  # TODO
        self.q_net = tf.keras.Sequential(layers)

    def call(self, inputs):
        """ 
        Return the ELBO for the current model 
        Hint: make a call to self.q_net(inputs).
        """
        return elbo(inputs, self.q_net(inputs), self.logits, self.means, self.sigmas)  # TODO

class VGMM_nu(tf.keras.Model):
    def __init__(self, d, k, nhidden_layers=3, hidden_size=256):
        super(VGMM_nu, self).__init__()
        """
        Hint: it helps to initialize variables close to zero with a small range 
        (about 0.1 standard deviation).

        self.logits = tf.Variable( TODO , name='logits')
        self.means = tf.Variable( TODO, name='means')
        self.sigmas = tf.Variable( TODO, name='sigmas')
        """
        self.logits = tf.Variable( tf.constant([2.197, 0.0], dtype=tf.float32) , name='logits', trainable=False)  # TODO
        self.means = tf.Variable( tf.random.normal(mean=0., stddev=0.1, shape=(k, 1, d)) , name='means')  # TODO
        self.sigmas = tf.Variable( tf.random.uniform(shape=(k, 1)) , name='sigmas')  # TODO
        
        """
        Setup a neural network with nhidden_layers with hidden_size number of units
        to output logits of approximate posteriors below.
        Hint: use tf.keras.layers.Dense and a ELU activation where appropriate.
        layers = [...]
        """
        layers = [tf.keras.layers.Dense(hidden_size, activation='elu') for i in range(nhidden_layers)] + [tf.keras.layers.Dense(k)]  # TODO
        self.q_net = tf.keras.Sequential(layers)

    def call(self, inputs):
        """ 
        Return the ELBO for the current model 
        Hint: make a call to self.q_net(inputs).
        """
        return elbo(inputs, self.q_net(inputs), self.logits, self.means, self.sigmas)  # TODO

def loss(model, inputs):
    return tf.reduce_mean(model(inputs))

def grad(model, inputs):
    with tf.GradientTape() as tape: 
        loss_value = loss(model, inputs)
    return tape.gradient(loss_value, model.trainable_variables)

def train(training_inputs):
    K = 2 
    d = training_inputs.shape[1]
    model = VGMM_nu(d, K, nhidden_layers=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    steps = 5000
    for i in range(steps):
        """ 
        Hint: 
        grads = something with training_inputs...
        """
        grads = grad(model, training_inputs)  # TODO
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 100 == 0:
            print("Loss at step {:03d}: {:.5f}".format(i, loss(model, training_inputs)))
    
    print(f"Model Logits: {model.logits}")
    print(f"Model Means: {model.means}")
    print(f"Model Std Dev: {model.sigmas}")

    return model

# If this is main file run the training
if __name__ == "__main__":
    # Load the data
    train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = load_data()

    model = train(tf.convert_to_tensor(train_inputs))
