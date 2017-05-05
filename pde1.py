#!/bin/python

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

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
        self.z = tf.placeholder(tf.float32, shape=[])
        self.dz = tf.placeholder(tf.float32, shape=[])

        w = [
            model_variable([2, 10], 'w1'),
            model_variable([10, 10], 'w2'),
            model_variable([10, 1], 'w3')
        ]

        b = [
            model_variable([10], 'b1'),
            model_variable([10], 'b2'),
            model_variable([1], 'b3')
        ]    

        def multilayer_perceptron(x, y, weights, biases):
            # input for 3D

            inputs = tf.expand_dims(tf.stack([x,y]), 0)
            # inputs = tf.stack([x,y])
            # Hidden layer with sigmoid activation
            layer_1 = tf.add(tf.matmul(inputs, w[0]), b[0])
            layer_1 = tf.nn.sigmoid(layer_1)
            # Hidden layer with sigmoid activation
            layer_2 = tf.add(tf.matmul(layer_1, w[1]), b[1])
            layer_2 = tf.nn.sigmoid(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, w[2]) + b[2]
  
            return out_layer

  
        NNout = multilayer_perceptron(self.x, self.y, w, b)
        NNout = tf.reshape(NNout, [])

        A = (1-self.x)*self.y**3 + self.x*(1+self.y**3)*np.exp(-1) + \
            (1-self.y)*self.x*(tf.exp(-self.x)-np.exp(-1)) + self.y*( (1+self.x)*np.exp(-1) - (1-self.x-2*self.x*np.exp(-1)) )
        
        self.zhat = A + self.x*(1-self.x)*self.y*(1-self.y)*NNout
        grad_x = tf.gradients(self.zhat, [self.x])
        grad_y = tf.gradients(self.zhat, [self.y])

        self.dzhat = tf.add(tf.gradients(grad_x[0], [self.x]), tf.gradients(grad_y[0], [self.y]))
        self.dzhat = self.dzhat
        
        self.mse = tf.reduce_mean(tf.pow(self.dzhat - (self.dz), 2))
        
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.mse + self.lambduh*self.l2_penalty
        
        self.error = tf.reduce_sum(tf.abs((self.zhat - self.z)/self.z))


    def train_init(self):
        model_variables = tf.get_collection('model_variables')            
        self.optim = (
            tf.train.AdamOptimizer(learning_rate=0.1)
            .minimize(self.loss, var_list=model_variables)
            )
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, x, y, z, dz):
        loss, mse, l2_penalty, _ = self.sess.run([self.loss, self.mse, self.l2_penalty, self.optim],
                                          feed_dict={self.x : x, self.y : y, self.z : z, self.dz : dz})
        print('loss: {}, mse: {}, l2_penalty {}'.format(loss, mse, l2_penalty))

    def train(self):
        for _ in range(self.nEpochs):
            # training set is 30
            for x, y, z, dz in self.data():
                self.train_iter(x, y, z, dz)

    def infer(self, x, y):
        return self.sess.run([self.x, self.y, self.zhat, self.dzhat], feed_dict={self.x : x, self.y : y})

    def eval(self, test):
        self.testresults = []
        print("Testing Accuracy:")
        for test in test:
            pointx, pointy, error = sess.run([self.x, self.y, self.error], feed_dict={self.x: test[0],
                                      self.y: test[1], self.z : test[2], self.dz : test[3]})
            print('input: ({},{}) , error: {}'.format(pointx, pointy, error))
            self.testresults.append((pointx,pointy,error))

# problem 5 in paper

def data():
    x = []
    # [x.append[i] for i in range(-2, 0.1, 2)]
    # num_samp = 10\
    xv = np.linspace(0, 1, 20)
    yv = np.linspace(0, 1, 20)    
    # xx, yy = np.meshgrid(x, y)
    for x in xv: 
        for y in yv: 
            # true solution:
            z = np.exp(-x)*(x+y**3)
            # assume Dirichlet BCs
            dz = np.exp(-x)*(x-2 + y**3 + 6*y)
            yield x, y, z, dz

sess = tf.Session()
model = Model(sess, data, nEpochs=100, learning_rate=1e-2, lambduh=1e-4)
model.train_init()
model.train()

test = []

for x, y, z, dz in data():
    test.append((x, y, z, dz))

model.eval(test)
    
# print(sess.run(tf.get_collection('model_variables')))

# generate manifold and plot

#debugger
# pdb.Pdb().set_trace()

x= np.linspace(0, 1, 20)
y= np.linspace(0, 1, 20)


test = []
for a in x:
    for b in y:
        test.append(model.infer(a, b))

test = np.array(test) 

xexamples, yexamples, zexamples, targets = zip(*list(data()))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# fig.set_size_inches(5, 3)
# x = [x[0] for x in test]
# y = [x[1] for x in test]
z = [x[2] for x in test]

xx, yy = np.meshgrid(x, y)
z = np.reshape(z, np.shape(xx))


print(np.shape(zexamples))
print(np.shape(xx))



zexamples = np.reshape(zexamples, np.shape(xx))
# plt.scatter(x, np.array(xexamples), np.array(yexamples), np.array(zexamples))
# ax.plot_surface(xx, yy, z)
ax.plot_surface(xx, yy, zexamples, label='actual')
ax.plot_surface(xx, yy, z, label='neural net')
# plt.xlim([-1, 2])
# plt.ylim([-10, 25])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# plt.title('Regression')
plt.tight_layout()
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')


# print(np.shapez(model.testresults))

# ax = fig.add_subplot(122,projection='3d')


# # xval = [x[0] for x in model.testresults]
# # yval = [x[1] for x in model.testresults]
# error = [x[2] for x in model.testresults]

# # xx, yy = np.meshgrid(xval, yval)
# # error = np.reshape(error, np.shape(xval))

# x= np.linspace(0, 1, 10)
# y= np.linspace(0, 1, 10)

# xx, yy = np.meshgrid(x, y)
# error = np.reshape(error, np.shape(xx))

# ax.plot_surface(x, y, error)
# # print(model.testresults)


# # plt.xlim([0, 1])
# # plt.ylim([-10, 25])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('error of z (%)')
# plt.title('Error Plot, AvgError: {:.2f}%'.format(np.mean(error)))
# plt.tight_layout()
# plt.savefig('errorplot.pdf', format='pdf', bbox_inches='tight')

plt.show()

plt.close()


