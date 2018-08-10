import tensorflow as tf
import numpy as np
from accuracy import *
from DataReader import *
from DRN import *
import os
from dice import *

tf.reset_default_graph()
directory = os.getcwd()

train_file = os.path.join(directory, 'data', 'train.txt')
test_file =  os.path.join(directory, 'data', 'val.txt')
logs = os.path.join(directory, 'logs')
trainloss = os.path.join(logs, 'train_loss.txt')


if os.path.isdir(logs) == False:
  os.makedirs(logs)

# choose network, can be either DRN18 or DRN26
network = 'DRN26'
# set parameters
batch_size=8
num_epochs=100
use_weights = 1
num_classes = 5
image_dims=[500,500,3]

data = DataReader(directory, batch_size, num_epochs, use_weights=1)
train_data = data.train_batch(train_file)
num_train_images = data.num_images

test_data = data.test_batch(test_file)
num_val_images = data.num_images

# determine number of iterations based on number of images
training_iterations = int(np.floor(num_train_images/batch_size))
validation_iterations = int(np.floor(num_val_images/batch_size))

handle = tf.placeholder(tf.string, shape=[])
# create iterator allowing us to switch between datasets
iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
next_element = iterator.get_next()
training_iterator = train_data.make_initializable_iterator()
val_iterator = test_data.make_initializable_iterator()

# create placeholder for train or test
train_network = tf.placeholder(tf.bool, [])

# get images and pass into network
image, label, weight = next_element
drn = DRN(image, image_dims, batch_size, num_classes, train_network, network)

# get predictions and logits
prediction = drn.pred
logits = drn.prob
label = tf.squeeze(label, 3)

# resize the logits using bilinear interpolation
imsize = tf.constant([iamge_dims[0], image_dims[1]], dtype=tf.int32)
logits = tf.image.resize_bilinear(logits, imsize)
prediction = tf.argmax(logits, 3)
print('Resized shape is {}'.format(logits.get_shape()))

 
# compute loss
if use_weights == 1:
  label_one_hot = tf.one_hot(label, num_classes)
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=logits)
  loss = loss*tf.squeeze(weight, 3)
else:
  # use sparse with flattened labelmaps
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(loss)

#loss = dice(logits, label, num_classes, use_weights=1)

# create summary
tf.summary.scalar('loss', loss)
tf.summary.image('images', image)
tf.summary.image('predictions', tf.cast(tf.expand_dims(prediction, 3), dtype=tf.uint8))
tf.summary.image('labels', tf.cast(tf.expand_dims(label, 3), dtype=tf.uint8))


# add weights with of first layer
print(tf.trainable_variables())
for var in tf.trainable_variables():
  if'_1' in var.name:
    tf.summary.histogram(var.name, var)
#tf.summary.scalar('learning_rate', learning_rate)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logs, tf.get_default_graph())

# global step to keep track of iterations
global_step = tf.Variable(0, trainable=False, name='global_step')

# create placeholder for learning rate
learning_rate = tf.placeholder(tf.float32, shape=[])

# training 
training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

saver = tf.train.Saver(max_to_keep=3)
init = tf.global_variables_initializer()

plt.figure(figsize=(18, 16))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  training_handle = sess.run(training_iterator.string_handle())
  validation_handle = sess.run(val_iterator.string_handle())
  
  sess.run(training_iterator.initializer)
  # initialize variables 
  sess.run(init)
  
  # check if checkpiont exists
  ckpt = tf.train.get_checkpoint_state(logs)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path) 
    print('Restoring session at step {}'.format(global_step.eval()))
    
  iteration = global_step.eval()
  current_epoch = int(np.ceil(iteration/training_iterations)) 
  
  while current_epoch < num_epochs:
    print(training_iterations)
    current_epoch = int(np.ceil(iteration/training_iterations)) 
    i = 1
    while i <= training_iterations:
      _, l = sess.run([training, loss], feed_dict={handle:training_handle, learning_rate:0.0001, train_network:True})
      iteration = global_step.eval()
      i = iteration - ((current_epoch-1)*training_iterations) 

      if iteration % 10 == 0:
        print('Training loss Epoch {} step {}, {} :{}'.format(current_epoch, iteration, i, l))
        # write loss to file
        with open(trainloss, 'a') as f: 
          f.write("Epoch: {} Step: {} Loss: {}\n".format(current_epoch, iteration, l))
          f.close()

      if (iteration %250 == 0) and (iteration > 0):
        # write summary to file
        summary = sess.run(merged,  feed_dict={handle:training_handle})
        train_writer.add_summary(summary, iteration)
        # save session
        print('Saving session at epoch {} step: {}'.format(current_epoch, iteration))
        saver.save(sess, logs + '/model.ckpt', global_step)

        
    sess.run(val_iterator.initializer)
    total_loss = 0
    for i in range(validation_iterations):
      print('validation step: {}'.format(i))
      img, lbl, wgt = next_element
      val_loss, img, lbl, wgt, pred = sess.run([loss, img, lbl, wgt, prediction], feed_dict={handle:validation_handle, train_network:False})
      total_loss += val_loss
      lbl = np.squeeze(lbl, 3)
      # evaluate accuracy
      accuracy = Jaccard(lbl, pred, num_classes)
      dice_score = DICE(lbl, pred, num_classes)

      if (i % 100 == 0) and (i >0):
        plt.subplot(131)
        img_temp = (img + abs(img.min()))/((abs(img.min()) + img.max()))
        plt.imshow(img_temp[0, :, :, :])
        plt.subplot(132)
        plt.imshow(np.squeeze(lbl[0, :, :]))
        # view prediction
        plt.subplot(133)
        plt.imshow(pred[0, :, :])
        plt.show()

        print(accuracy)
        print(dice_score)
        
f.close()