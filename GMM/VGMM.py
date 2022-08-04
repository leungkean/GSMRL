import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm

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

def log_multivariate_normal(x, mean, sigma):
    """ Compute log multivariate normal for k classes
    Args:
        x: N x d tensor input.
        mean: k x 1 x d tensor of means.
        sigma: k x 1 x d tensor of log std. dev.
    Return:
        log_prob: N x k tensor of log probabilities.
    """
    sigma_det = tf.reshape(tf.math.reduce_sum(tf.math.log(sigma), axis=2), (1, mean.shape[0]))
    sigma_inv = tf.reshape(tf.linalg.diag(1/sigma), (sigma.shape[0], sigma.shape[2], sigma.shape[2]))
    x_mean = x - mean
    k = mean.shape[0]
    const = -0.5*k*tf.math.log((2*np.pi)) - 0.5*sigma_det
    return const - 0.5*tf.transpose(tf.math.reduce_sum(tf.math.multiply(tf.matmul(x_mean, sigma_inv), x_mean), axis=2))

def elbo(inputs, q_logits, logits, means, sigmas):
    """ Compute the ELBO for a given GMM model and approximate posterior.
    Args:
        inputs: N x d points to compute ELBO at.
        q_logits: N x ncomp tensor of logits for approximate posterior of z_i.
        logits: ncomp tensor of mixing priors of mixture model
        means: ncomp x 1 x d tensor of means of mixture model.
        sigmas: ncomp x d x d tensor of log std. dev. of mixture model.
    Return:
        kl: N tensor of KL(p_i || q_i)
    """
    """
    Hint: Do not try to use a reparameterization-type trick. Instead, directly 
    compute any expectations.
    """
    norm_q = tf.nn.softmax(q_logits)
    decode = log_multivariate_normal(inputs, means, sigmas)
    new_logits = tf.repeat(tf.expand_dims(logits,axis=0), repeats=[inputs.shape[0]], axis=0)
    return tf.math.reduce_sum(norm_q*decode, axis=1) - kl_divergence(q_logits, new_logits)  # TODO

class VGMM(tf.keras.Model):
    def __init__(self, d, k, nhidden_layers=3, hidden_size=256):
        super(VGMM, self).__init__()
        """
        Hint: it helps to initialize variables close to zero with a small range 
        (about 0.1 standard deviation).

        self.logits = tf.Variable( TODO , name='logits')
        self.means = tf.Variable( TODO, name='means')
        self.sigmas = tf.Variable( TODO, name='sigmas')
        """
        self.logits = tf.Variable( tf.random.normal(mean=0., stddev=0.1, shape=(k,)) , name='logits')  # TODO
        #self.logits = tf.Variable( tf.constant([2.2, 0.0], dtype=tf.float32) , name='logits', trainable=False)  # TODO
        self.means = tf.Variable( tf.random.normal(mean=0., stddev=0.1, shape=(k, 1, d)) , name='means')  # TODO
        self.sigmas = tf.Variable( tf.random.uniform(shape=(k, 1, d)) , name='sigmas', constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-4, clip_value_max=40.0))  # TODO
        
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
    return -tf.reduce_mean(model(inputs))

def grad(model, inputs):
    with tf.GradientTape() as tape: 
        loss_value = loss(model, inputs)
    return tape.gradient(loss_value, model.trainable_variables)

def train(training_inputs):
    K = 2 
    d = inputs.shape[1]
    model = VGMM(d, K, nhidden_layers=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    steps = 10000
    for i in tqdm(range(steps)):
        """ 
        Hint: 
        grads = something with training_inputs...
        """
        grads = grad(model, training_inputs)  # TODO
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 500 == 0:
            print("Loss at step {:03d}: {:.5f}".format(i, loss(model, training_inputs)))
    
    np.save('logits.npy', model.logits.numpy())
    np.save('means.npy', tf.reshape(model.means, shape=(K, d)).numpy())
    np.save('sigmas.npy', tf.reshape(model.sigmas, shape=(K, d)).numpy())

    print(f"Model Logits: {model.logits}")
    print(f"Model Means: {model.means}")
    print(f"Model Std Dev: {model.sigmas}")

    return model

def make_data(N, logits, means, sigmas): 
    z = tf.transpose(tf.random.categorical([logits], N)) 
    y = tf.random.normal((N, 1))

    x = tf.gather(means, z)+tf.gather(tf.math.exp(sigmas), z)*y 
    return x

# If this is main file run the training
if __name__ == "__main__":
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
    inputs = np.concatenate((cheap_feat, exp_feat), axis=1)

    labels = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
    labels = labels.reshape(-1, 1)

    #inputs = np.concatenate((inputs, labels), axis=1)

    train_indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1] == 0
    train_inputs = inputs[train_indices, :]

    model = train(tf.convert_to_tensor(train_inputs))
