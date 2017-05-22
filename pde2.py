#!/bin/python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
import scipy.optimize


    
class Model():
    def __init__(self, sess, data, nEpochs, learning_rate, lambduh, batch_size):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.learning_rate = learning_rate
        self.lambduh = lambduh
        self.batch_size = batch_size
        self.build_model()
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None])
        self.y = tf.placeholder(tf.float32, shape=[None])
        self.z = tf.placeholder(tf.float32, shape=[None])
        self.dz = tf.placeholder(tf.float32, shape=[None])

        # tf.slim
        def NeuralNet(x, y):
            with tf.variable_scope('model', reuse=False):
                # input
                #inputs = tf.expand_dims(tf.stack([x,y]), 0)
                inputs = tf.transpose([x,y])
                # inputs = tf.stack([x,y])
                # NOTE: elu activation function on 3 layers returns NaN losses ??

                net = slim.fully_connected(
                    inputs,
                    10,
                    activation_fn=tf.nn.sigmoid,
                    weights_regularizer=slim.l2_regularizer(self.lambduh),
                    variables_collections=['model'],
                    scope='fc1')
                net = slim.fully_connected(
                    net,
                    10,
                    activation_fn=tf.nn.sigmoid,
                    weights_regularizer=slim.l2_regularizer(self.lambduh),
                    variables_collections=['model'],
                    scope='fc2')
                # layer1 = tf.nn.lrn(layer1)
                net = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(self.lambduh),
                    variables_collections=['model'],
                    scope='out')
                return net
  
        # NNout = multilayer_perceptron(self.x, self.y, w, b)
        NNout = NeuralNet(self.x, self.y)
        NNout = tf.squeeze(NNout)


        self.zhat = self.y*tf.sin(np.pi*self.x) + self.x*(self.x - 1)*self.y*(self.y-1)*NNout


        # grad_x = tf.gradients(self.zhat, [self.x])
        # grad_y = tf.gradients(self.zhat, [self.y])

        # self.dzhat = tf.add(tf.gradients(grad_x[0], [self.x]), tf.gradients(grad_y[0], [self.y]))

        # derivatives of Neural Net
        NN_x = tf.gradients(NNout, [self.x])
        NN_y = tf.gradients(NNout, [self.y])
        
        NN_x2 = tf.gradients(NN_x[0], [self.x])
        NN_y2 = tf.gradients(NN_y[0], [self.y])

        # x and y terms of gradient
        grad_x2 = -tf.square(np.pi)*self.y*tf.sin(np.pi*self.x) + \
                (tf.square(self.y)-self.y)*( (tf.square(self.x)-self.x)*NN_x2[0] + (4*self.x - 2)*NN_x[0] + 2*NNout)

        grad_y2 = (tf.square(self.x)-self.x)*( (tf.square(self.y)-self.y)*NN_y2[0] + (4*self.y-2)*NN_y[0] + 2*NNout)

        # predicted gradient
        self.dzhat = tf.add(grad_x2, grad_y2)
        
        self.mse = tf.reduce_sum(tf.square((self.dzhat - self.dz)))
        self.l2_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
        self.loss = self.mse + self.lambduh*self.l2_penalty
        
        self.error = tf.reduce_sum(tf.square(self.zhat - self.z))


    def train_init(self):   
        self.optim = (
            tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            .minimize(self.loss, var_list=tf.get_collection(key='model'))
            )
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, x, y, z, dz):
        loss, mse, l2_penalty, _ = self.sess.run([self.loss, self.mse, self.l2_penalty, self.optim],
                                          feed_dict={self.x : x, self.y : y, self.z : z, self.dz : dz})
        print('loss: {}, mse: {}, l2_penalty {}'.format(loss, mse, l2_penalty))
        return loss

    def train(self):
        self.loss_tracker = []
        for _ in range(self.nEpochs):
            # training set is 30
            loss = 0
            count = 0
            batch = [[],[],[],[]];
            for x, y, z, dz in self.data():
                batch[0].append(x);
                batch[1].append(y);
                batch[2].append(z);
                batch[3].append(dz);
                count += 1;
                if (count == self.batch_size):
                    loss += self.train_iter(batch[0], batch[1], batch[2], batch[3])
                    count = 0
                    batch = [[],[],[],[]];
            self.loss_tracker.append(loss/len(list(self.data())))

    def infer(self, x, y):
        return self.sess.run([self.x, self.y, self.zhat, self.dzhat], feed_dict={self.x : [x], self.y : [y]})

    def eval(self, test):
        self.testresults = []
        print("Testing Accuracy:")
        for test in test:
            pointx, pointy, error = sess.run([self.x, self.y, self.error], feed_dict={self.x: [test[0]],
                                      self.y: [test[1]], self.z : [test[2]], self.dz : [test[3]]})
            print('input: ({},{}) , error: {}'.format(pointx, pointy, error))
            self.testresults.append((pointx,pointy,error))

# laplaces equation

def data():
    x = []
    # [x.append[i] for i in range(-2, 0.1, 2)]
    # num_samp = 10\
    xv = np.linspace(0, 1, 50)
    yv = np.linspace(0, 1, 50)    
    # xx, yy = np.meshgrid(x, y)
    for x in xv: 
        for y in yv: 
            # true solution:
            z = np.sin(np.pi*x)*np.sinh(np.pi*y)/(np.sinh(np.pi))
            # assume Dirichlet BCs
            dz = 0
            yield x, y, z, dz

sess = tf.Session()
model = Model(sess, data, nEpochs=120, learning_rate=1e-2, lambduh=1e-3, batch_size = 40)
model.train_init()
model.train()

test = []

for x, y, z, dz in data():
    test.append((x, y, z, dz))

model.eval(test)
    

x= np.linspace(0, 1, 50)
y= np.linspace(0, 1, 50)


test = []
for a in x:
    for b in y:
        test.append(model.infer(a, b))

test = np.array(test) 

xexamples, yexamples, zexamples, targets = zip(*list(data()))

fig = plt.figure()
ax = fig.add_subplot(211,projection='3d')
ay = fig.add_subplot(212,projection='3d')
# fig.set_size_inches(5, 3)
# x = [x[0] for x in test]
# y = [x[1] for x in test]
z = [x[2] for x in test]

xx, yy = np.meshgrid(x, y)
z = np.reshape(z, np.shape(xx))


zexamples = np.reshape(zexamples, np.shape(xx))
# plt.scatter(x, np.array(xexamples), np.array(yexamples), np.array(zexamples))
# ax.plot_surface(xx, yy, z)
ax.plot_surface(xx, yy, zexamples, label='actual')
ay.plot_surface(xx, yy, z, label='neural net')
# plt.xlim([-1, 2])
# plt.ylim([-10, 25])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (Actual)')
ay.set_xlabel('x')
ay.set_ylabel('y')
ay.set_zlabel('z (Neural Net)')
plt.suptitle('Actual vs. Neural Net Solution')
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')


fig = plt.figure()
ay = fig.add_subplot(111)
x = []
[x.append(i) for i in range(len(model.loss_tracker))]
ay.set_xlabel('batch number')
ay.set_ylabel('MSE Loss')
plt.suptitle('Loss during training')
plt.plot(x, model.loss_tracker)



fig = plt.figure()
bx = fig.add_subplot(111,projection='3d')


# # xval = [x[0] for x in model.testresults]
# # yval = [x[1] for x in model.testresults]
error = [x[2] for x in model.testresults]

# # xx, yy = np.meshgrid(xval, yval)
# # error = np.reshape(error, np.shape(xval))

x= np.linspace(0, 1, 50)
y= np.linspace(0, 1, 50)

xx, yy = np.meshgrid(x, y)
error = np.reshape(error, np.shape(xx))


bx.plot_surface(xx, yy, error)


# plt.xlim([0, 1])
# plt.ylim([-10, 25])
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('error of z')
plt.title('Error Plot, MSE: {:.7f}, Max Error: {:.7f}'.format(np.mean(np.absolute(error)) , 
                                                                        np.max(np.absolute(error))))
plt.tight_layout()
plt.savefig('errorplot.pdf', format='pdf', bbox_inches='tight')

plt.figure()
cp = plt.contourf(xx, yy, error)
plt.title('Error Plot, MSE: {:.7f}, Max Error: {:.7f}'.format(np.mean(np.absolute(error)) , 
                                                                        np.max(np.absolute(error))))
plt.xlabel('x')
plt.ylabel('y')


plt.show()

plt.close()


