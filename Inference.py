import tensorflow as tf
import numpy as np
import os
from PIL import Image as im
from DataReader import DataReader
from DRN import DRN
from accuracy import *

tf.reset_default_graph()

directory = os.getcwd()
data_file = os.path.join(directory, 'data' , 'val.txt')
logs = os.path.join(directory, 'logs')
pred_directory = os.path.join(directory, 'Pred')

# make directory for storing predictions if it does not exist
if os.path.isdir(pred_directory) == False:
  os.makedirs(pred_directory)

# choose network, can be either DRN18 or DRN26
network = 'DRN26'
# set parameters
batch_size=8
num_epochs=100
use_weights = 1
num_classes = 5
image_dims=[500,500,3]

data = DataReader(directory, batch_size, num_epochs, use_weights=0)
dataset = data.test_batch(data_file)
num_images = data.num_images

# get image filenames
image_list = data.image_list

# determine number of iterations based on number of images
num_iterations = int(np.floor(num_images/batch_size))

# create iterator allowing us to switch between datasets
data_iterator = dataset.make_one_shot_iterator()
next_element = data_iterator.get_next()

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
imsize = tf.constant([image_dims[0], image_dims[1]], dtype=tf.int32)
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

# global step to keep track of iterations
global_step = tf.Variable(0, trainable=False, name='global_step')

saver = tf.train.Saver(max_to_keep=3)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
 
  # initialize variables 
  sess.run(init)
  
  # check if checkpiont exists
  ckpt = tf.train.get_checkpoint_state(logs)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path) 
    print('Restoring session at step {}'.format(global_step.eval()))
    
  iteration = global_step.eval()
  lumen_IOU = []
  plaque_IOU = []
  for i in range(num_iterations):
    print('step: {} of {}'.format(i, num_iterations))
    val_loss, img, lbl, wgt, pred = sess.run([loss, img, lbl, wgt, prediction], feed_dict={train_network:False})
    total_loss += val_loss
    lbl = np.squeeze(lbl, 3)
    # evaluate accuracy
    accuracy = Jaccard(lbl, pred, num_classes)
    dice_score = DICE(lbl, pred, num_classes)
    lumen_IOU.append(accuracy[2])
    plaque_IOU.append(accuracy[3])

    fnames = image_list[i]
    # write images to file
    for j in range(pred.shape[0]):
      _, dir, fname = fnames[j].split('/')
	  # drop file extension
      fname = fname.split('.')[0]
      # make case directory if doesn't exist
      case_directory = pred_directory + '/' + dir
      if os.path.isdir(case_directory) == False:
        os.makedirs(case_directory)

      img_write = im.fromarray(pred[j, :, :], "L")
      img_write.save(os.path.join(case_directory, "pred_image" + str(i)+".png"))
	
  print('Total lumen IOU is {}'.format(np.mean(lumen_IOU)))
  print('Total plaque IOU is {}'.format(np.mean(plaque_IOU)))