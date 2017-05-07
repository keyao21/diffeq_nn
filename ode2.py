#!/bin/python

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.contrib.slim as slim

# need to change collocation method s.t ranges in domain that have high error will be sampled more
# implement partial diff eq


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

        # w = [
        #     model_variable([10, 10], 'w1'),
        #     model_variable([10, 10], 'w2'),
        #     model_variable([10, 1], 'w3')
        # ]

        # b = [
        #     model_variable([10], 'b1'),
        #     model_variable([10], 'b2'),
        #     model_variable([1], 'b3')
        # ]    

        # def multilayer_perceptron(x, weights, biases):
        #     # Hidden layer with sigmoid activation
        #     layer_1 = tf.add(tf.matmul(x, w[0]), b[0])
        #     layer_1 = tf.nn.sigmoid(layer_1)

        #     # Hidden layer with sigmoid activation
        #     # layer_2 = tf.add(tf.matmul(layer_1, w[1]), b[1])
        #     # layer_2 = tf.nn.sigmoid(layer_2)

        #     # layer_2 = tf.add(tf.matmul(layer_2, w[1]), b[1])
        #     # layer_2 = tf.nn.sigmoid(layer_2)

        #     # layer_2 = tf.add(tf.matmul(layer_2, w[1]), b[1])
        #     # layer_2 = tf.nn.sigmoid(layer_2)

        #     # layer_2 = tf.add(tf.matmul(layer_2, w[1]), b[1])
        #     # layer_2 = tf.nn.sigmoid(layer_2)


        #     # Output layer with linear activation
        #     out_layer = tf.matmul(layer_1, w[2]) + b[2]
  
        #     return out_layer


        def mlp_slim(x):
            # x = slim.layers.flatten(x, scope='flatten3')
            x = tf.expand_dims([x], 0)

            with tf.variable_scope('model', reuse=False):
                x = slim.layers.fully_connected(x, 
                    10, 
                    activation_fn=tf.nn.relu, 
                    variables_collections=['model'], 
                    weights_regularizer=slim.l2_regularizer(0.01),
                    scope='fc/fc_1'
                    )
                x = slim.layers.fully_connected(x, 
                    10, 
                    activation_fn=tf.nn.relu, 
                    variables_collections=['model'], 
                    weights_regularizer=slim.l2_regularizer(0.01),
                    scope='fc/fc_2'
                    )
                x = slim.layers.fully_connected(x, 
                    10, 
                    activation_fn=tf.nn.sigmoid, 
                    variables_collections=['model'], 
                    weights_regularizer=slim.l2_regularizer(0.01),
                    scope='fc/fc_3'
                    )
                x = slim.layers.fully_connected(x, 
                    1, 
                    activation_fn=None, 
                    variables_collections=['model'], 
                    weights_regularizer=slim.l2_regularizer(0.01),
                    scope='out')
                return x
  
        NNout = mlp_slim(self.x)
        
        # NNout = mlp_slim(self.x)

        NNout = tf.reshape(NNout, [])

        self.yhat = NNout*self.x

        self.dyhat = NNout + self.x*tf.gradients(NNout, self.x)
        
        self.mse = tf.reduce_mean(tf.pow(self.dyhat - self.dy, 2))
        
        self.l2_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
        self.loss = self.mse + self.lambduh*self.l2_penalty
        
        self.accuracy = tf.abs(self.yhat - self.y)


    def train_init(self):
        model_variables = tf.get_collection(key='model')            
        self.optim = (
            tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            .minimize(self.loss, var_list=model_variables)
            )
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, x, y, dy):
        loss, mse, l2_penalty, _ = self.sess.run([self.loss, self.mse, self.l2_penalty, self.optim],
                                          feed_dict={self.x : x, self.y : y, self.dy : dy})
        print('loss: {:.5f}, mse: {:.5f}, l2_penalty {:.2f}'.format(loss, mse, l2_penalty))
        return loss

    def train(self):
        self.loss_tracker = []
        for _ in range(self.nEpochs):
            # training set is 30
            loss = 0
            for x, y, dy in self.data():
                loss += self.train_iter(x, y, dy)
            self.loss_tracker.append(loss/len(list(self.data())))
       
    def infer(self, x):
        return self.sess.run([self.x, self.y, self.yhat],feed_dict={self.x : x, self.y: y})

    def eval(self, test):
        self.testresults = []
        print("Testing Accuracy:")
        for test in test:
            x, y, acc = sess.run([self.x, self.y, self.accuracy], feed_dict={self.x: test[0],
                                      self.y: test[1], self.dy : test[2]})
            print('input: {}, accuracy: {}'.format(x, acc))
            self.testresults.append((x,y,acc))

#let dy/dx = exp(-x/5)cos(x) - 1/(5y)
#y = exp(-x/5)sin(x)

def data():
    xv = np.linspace(0,2,10)
    # add = np.linspace(1,2,5)
    # xv = np.concatenate((xv, add))
    for x in xv:
        # y = (3/25)*(-5*x + np.exp(5*x)-1)
        y = np.exp(-x/5)*np.sin(x)
        #let y(0) = 0
        # dy = 5*y + 3*x
        dy = np.exp(-x/5)*np.cos(x) - y/(5)
        yield x, y, dy
    

sess = tf.Session()
model = Model(sess, data, nEpochs=50, learning_rate=1e-1, lambduh=1e-4)
model.train_init()
model.train()

test = []

for x, y, dy in data():
    test.append((x, y, dy))

model.eval(test)
    
# print(sess.run(tf.get_collection('model_variables')))

# generate manifold and plot

x= np.linspace(0,2,30)

test = []
for a in x:
    test.append(model.infer(a))
test = np.array(test) 

xexamples, yexamples, targets = zip(*list(data()))


fig, ax = plt.subplots(1,1)
fig.set_size_inches(5, 3)
x = [x[0] for x in test]
y = [x[1] for x in test]
yhat = [x[2] for x in test]
plt.plot(x, yhat, '-',  x, yhat, '.',label='neural net')
plt.plot( np.array(xexamples), np.array(yexamples), '-', label='actual')
# plt.plot( x, y, '-', label='actual')

# plt.xlim([-1, 2])
# plt.ylim([-10, 25])
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y').set_rotation(0)
# plt.title('Regression')
plt.tight_layout()
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')



fig = plt.figure()
ay = fig.add_subplot(111)
x = []
[x.append(i) for i in range(len(model.loss_tracker))]
ay.set_xlabel('batch number')
ay.set_ylabel('MSE Loss')
plt.suptitle('Loss during training')
plt.plot(x, model.loss_tracker)
plt.savefig('loss.pdf', format='pdf', bbox_inches='tight')


# print(np.shapez(model.testresults))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 3)

yval = [x[0] for x in model.testresults]
error = [x[2] for x in model.testresults]
# plt.bar(yval, error)
plt.plot(yval, error)
# print("average error: " + np.mean(error))
# print('input: {}, accuracy: {}'.format(point, acc))

# plt.xlim([0, 1])
# plt.ylim([-10, 25])
ax.set_xlabel('x')
ax.set_ylabel('error').set_rotation(0)
plt.title('Error Plot, Average Error: {:.4f}'.format(np.mean(error)))
plt.tight_layout()
plt.savefig('errorplot.pdf', format='pdf', bbox_inches='tight')

plt.show()

