import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Empirical, Beta, Normal
import matplotlib.pyplot as plt
import numpy as np

##Probabilistic Grammar Inference using edward

##Grammar
def generateFormula(theta):

    def cond(theta, sentence):
        return tf.cast(Bernoulli(probs=theta), tf.bool)

    def body(theta, sentence):
        return theta, a + sentence

    sentence = tf.constant("B")
    a = tf.constant("A")
    return tf.while_loop(cond, body, loop_vars=[theta, sentence])[1]

theta = Beta(1., 1., name="theta")
formula = generateFormula(theta)

##Sampling:
with tf.Session() as sess:
    for i in range(10):
        print(generateFormula(.999).eval())
    print('\n')
   

##Observations

data = "AAAAAAAAAAAAAAAAAB"

##Infer:
T=10000
qtheta = Empirical(params=tf.Variable(0.5+tf.zeros([T])), name="qtheta") 
sess = ed.get_session()
inference = ed.HMC({theta: qtheta}, {formula: data})
inference.run()

##Results:
qtheta_samples = qtheta.sample(1000).eval()
print(qtheta_samples.mean())
