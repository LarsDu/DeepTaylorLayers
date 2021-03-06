"""
Implementation of a Convolutional Neural Network with DeepTaylor decomposition 

Based off the tutorial
http://tensorflow.org/tutorials/mnist/beginners/index.md

Written by Lawrence Du 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


sys.path.append("..")
from dtlayers import dtlayers as dtl
from dtlayers.formatplot import Formatter  #see z values of mouse-over


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('mode','train',"""Options: \'train\' or \'visualize\' """)

flags.DEFINE_string('data_dir',"mnist/data","""MNIST data directory""")
flags.DEFINE_string('save_dir','./mnist_example1_model',"""Directory under which to place checkpoints""")
flags.DEFINE_integer('num_iterations',10000,""" Number of training iterations """)
flags.DEFINE_integer('num_visualize',10,""" Number of samples to visualize""")
flags.DEFINE_integer('batch_size',25,""" Number of samples to visualize""")

def main(_):
   
    mode = FLAGS.mode
    num_visualize = FLAGS.num_visualize
    num_iterations = FLAGS.num_iterations
    
    save_dir = FLAGS.save_dir
    checkpoint_dir = save_dir+"/checkpoints"
    summary_dir =save_dir+"/summaries"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = dtl.ImageInput(x,image_shape=[28,28,1])
    
  
    #conv filter dimensions are  w,h,input_dims,output_dims
    
    #Example with only convolutional layers

    cl1 = dtl.Conv2d(x_image, filter_shape = [5,5,1,10],padding = 'VALID',name='conv1')
    r1 = dtl.Relu(cl1)
    p1 = dtl.AvgPool(r1,[2,2],'avg_pool1')
    cl2 = dtl.Conv2d(p1, filter_shape = [5,5,10,25],
                       strides=[1,1,1,1],
                       padding='VALID',name='conv2')
    r2 = dtl.Relu(cl2)
    p2 = dtl.AvgPool(r2,[2,2],'avg_pool2')
    flat = dtl.Flatten(p2)
    cl3 = dtl.Conv2d(p2, filter_shape = [4,4,25,100],
                       strides=[1,1,1,1],
                       padding='VALID',name='conv3')
    r3 = dtl.Relu(cl3)
    p3 = dtl.AvgPool(r3,[2,2],'avg_pool3')
    cl4 = dtl.Conv2d(p3,filter_shape=[1,1,100,10],
                      strides=[1,1,1,1],
                      padding='VALID',
                      name='conv4')
    flat = dtl.Flatten(cl4)

    nn = dtl.Network(x_image,[flat],bounds=[0.,1.])

    '''
    Note:
    	Initializing with only the last layer flat will automatically construct the
        list of layers:

        dtl.Network(x_image,[flat],bounds=[0.,1.])
    '''
    
    y = nn.forward() #These are the logits

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10]) #These are the labels

    
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        #Test conv op by itself
            #batch_xs, batch_ys = mnist.train.next_batch(1)
            #tconv = sess.run(cl1.output,feed_dict = {x:batch_xs})
            #print ("Tconv",np.asarray(tconv).shape)

        if mode == "train":
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ("Checkpoint restored")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint found on ", ckpt)

            print ("Running training mode...")
            #Instatiate SummaryWriter to output summaries and Graph (optional)
            summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)

            for i in range(num_iterations):
                if i%2000 == 0:
                    print (i,"training iterations passed")
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                #cl1_val = sess.run(cl1.output,feed_dict={x: batch_xs, y_: batch_ys})
                #print (np.asarray(cl1_val).shape)
                # Test trained model

                if i% 4000 == 0:
                    ckpt_name = "model_ckpt"
                    save_path =saver.save(sess,checkpoint_dir+os.sep+ckpt_name)
                    print("Model saved in file: %s" % save_path)
    
                
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                         y_: mnist.test.labels}))

            eval_model(sess,correct_prediction,mnist,x,y_)

            ckpt_name = "model_ckpt"
            save_path =saver.save(sess,checkpoint_dir+os.sep+ckpt_name)
            print("Model saved in file: %s" % save_path)

        #Decomposition:
        #Look up and retrieve most recent checkpoint
        elif mode =="visualize":
            print ("Checkpoint dir", checkpoint_dir)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ("Checkpoint restored")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint found on", ckpt.model_checkpoint_path)

        

            for i in range(num_visualize):
                batch_xs, batch_ys = mnist.train.next_batch(1)

                rj_final_op = tf.multiply(y,y_)
                r_input,rj_final_val = sess.run([nn.relevance_backprop(rj_final_op),rj_final_op],
                                   feed_dict={x:batch_xs,y_:batch_ys})
                
                r_input_img=np.squeeze(r_input) 
                
                r_input_sum = np.sum(r_input)

                print ("Rj final {}, Rj sum {}".format(np.sum(rj_final_val),r_input_sum))


                #utils.visualize(r_input[:,2:-2,2:-2],utils.heatmap,'deeptaylortest_'+str(i)+'_.png')
                
                #Display original input
                #plt.imshow(np.reshape(np.asarray(batch_xs),(28,28)))

                yguess_mat = sess.run(y,
                                 feed_dict={x: batch_xs, y_: batch_ys})
                yguess = yguess_mat.tolist()
                yguess = yguess[0].index(max(yguess[0]))
                actual = batch_ys[0].tolist().index(1.)

                print ("Guess:",(yguess))
                print ("Actual:",(actual))


                #Display relevance heatmap
                fig,ax = plt.subplots()
                im = ax.imshow(r_input_img,cmap=plt.cm.Reds,interpolation='nearest')
                ax.format_coord = Formatter(im)
                plt.show()
                
                #gradient=np.squeeze(np.asarray(sess.run(sqr_sensitivity,feed_dict={x:batch_xs})))
                #gradient = np.reshape(gradient,(28,28))
                #print (gradient.shape)
                #plt.imshow(gradient,cmap=plt.cm.Reds,interpolation='nearest')
                #plt.show()




def eval_model(sess,correct_prediction,mnist,x,y_):

    eval_batch_size=1
    num_examples = mnist.test.images.shape[0]
    #num_labels = mnist.test.labels.shape[0]
    steps_per_epoch = num_examples//eval_batch_size

    num_correct=0
    for _ in range(steps_per_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(eval_batch_size)
        feed_dict={x: batch_xs, y_: batch_ys}
        num_correct += sess.run(correct_prediction,feed_dict=feed_dict)

    precision = float(num_correct)/num_examples
    #summary_writer.add_summary(loss+'/'+summar_tag,step)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, num_correct, precision))

            
            
if __name__ == '__main__':
    tf.app.run()
