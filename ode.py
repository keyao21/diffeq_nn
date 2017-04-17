#!/bin/python

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import tensorflow.contrib.slim as slim


def model_variable(shape, name):
        variable = tf.get_variable(name=name,
                                   dtype=tf.float32,
                                   shape=shape,
                                   initializer=tf.random_normal_initializer(mean=0, stddev=2)
        )
        tf.add_to_collection('model_variables', variable)
        tf.add_to_collection('l2', tf.reduce_sum(tf.pow(variable,2)))
        return variable
    
class Model():
    def __init__(self, sess, data, nEpochs, learning_rate, lambduh):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.learning_rate = learning_rate
        self.lambduh = lambduh
        self.build_model()
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[])
        self.y = tf.placeholder(tf.float32, shape=[])
        self.dy = tf.placeholder(tf.float32, shape=[])

        w = [
            model_variable([1, 10], 'w1'),
            model_variable([10, 10], 'w2'),
            model_variable([10, 1], 'w3')
        ]

        b = [
            model_variable([10], 'b1'),
            model_variable([10], 'b2'),
            model_variable([1], 'b3')
        ]    

        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with sigmoid activation
            layer_1 = tf.add(tf.multiply(x, w[0]), b[0])
            layer_1 = tf.nn.sigmoid(layer_1)
            # Hidden layer with sigmoid activation
            layer_2 = tf.add(tf.matmul(layer_1, w[1]), b[1])
            layer_2 = tf.nn.sigmoid(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_1, w[2]) + b[2]
  
            return out_layer

  
        NNout = multilayer_perceptron(self.x, w, b)
        NNout = tf.reshape(NNout, [])

        self.yhat = NNout*self.x

        self.dyhat = NNout + self.x*tf.gradients(NNout, self.x)
        
        self.mse = tf.reduce_mean(tf.pow(self.dyhat - self.dy, 2))
        
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.mse + self.lambduh*self.l2_penalty
        
        self.accuracy = (self.dyhat - self.dy)/self.dy


    def train_init(self):
        model_variables = tf.get_collection('model_variables')            
        self.optim = (
            tf.train.GradientDescentOptimizer(learning_rate=0.1)
            .minimize(self.loss, var_list=model_variables)
            )
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, x, y, dy):
        loss, mse, l2_penalty, _ = self.sess.run([self.loss, self.mse, self.l2_penalty, self.optim],
                                          feed_dict={self.x : x, self.y : y, self.dy : dy})
        print('loss: {}, mse: {}, l2_penalty {}'.format(loss, mse, l2_penalty))

    def train(self):
        for _ in range(self.nEpochs):
            for x, y, dy in self.data():
                self.train_iter(x, y, dy)

    def infer(self, x):
        return self.sess.run([self.x, self.dyhat],feed_dict={self.x : x})

    def eval(self, test):
        self.testresults = []
        print("Testing Accuracy:")
        for test in test:
            point, acc = sess.run([self.x, self.accuracy], feed_dict={self.x: test[0],
                                      self.y: test[1], self.dy : test[2]})
            print('input: {}, accuracy: {}'.format(point, acc))
            self.testresults.append((point,acc))

#let dy/dx = exp(-x/5)cos(x) - 1/(5y)
#y = exp(-x/5)sin(x)

def data(low=-1, up=2, points=50):
    x = []
    # [x.append[i] for i in range(-2, 0.1, 2)]
    # num_samp = 10
    x = np.linspace(low,up,points)
    data = []
    for x in x:
        # y = (3/25)*(-5*x + np.exp(5*x)-1)
        y = np.exp(-x/5)*np.sin(x)
        #let y(0) = 0
        # dy = 5*y + 3*x
        dy = np.exp(-x/5)*np.cos(x) - 1/(5*y)
        yield x, y, dy

sess = tf.Session()
model = Model(sess, data, nEpochs=200, learning_rate=1e-2, lambduh=1e-4)
model.train_init()
model.train()

test = []

for x, y, dy in data():
    test.append((x, y, dy))

model.eval(test)
    
# print(sess.run(tf.get_collection('model_variables')))

# generate manifold and plot

x= np.linspace(-3, 4, 30)

test = []
for a in x:
    test.append(model.infer(a))
test = np.array(test) 
# print("hello: this is me")
# print(test)

examples, x, targets = zip(*list(data()))

fig, ax = plt.subplots(1,1)
fig.set_size_inches(5, 3)
x = [x[0] for x in test]
dy = [x[1] for x in test]
plt.plot(x, dy, 'o', x, dy, '-', np.array(examples), np.array(targets), '-')
# plt.xlim([-1, 2])
# plt.ylim([-10, 25])
ax.set_xlabel('x')
ax.set_ylabel('dy/dx').set_rotation(0)
# plt.title('Regression')
plt.tight_layout()
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.close()


# print(np.shapez(model.testresults))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 3)

yval = [x[0] for x in model.testresults]
error = [x[1] for x in model.testresults]
plt.plot(yval, error, 'or')
print(model.testresults)


# plt.xlim([0, 1])
# plt.ylim([-10, 25])
ax.set_xlabel('x')
ax.set_ylabel('error').set_rotation(0)
plt.title('error')
plt.tight_layout()
plt.savefig('errorplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

