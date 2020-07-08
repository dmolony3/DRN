import os
import tensorflow as tf
import numpy as np
from data_reader import DataReader
from drn import DRN
from PIL import Image as im

def predict(config):

  tf.reset_default_graph()

  directory = os.getcwd()

  pred_directory = os.path.join(directory, 'Pred')

  data = DataReader(config.directory, config.image_dims, config.batch_size, 
                    config.num_epochs, use_weights=False)
  dataset = data.test_batch(config.val_file)
  num_images = data.num_images

  # get image filenames
  image_list = data.image_list

  # determine number of iterations based on number of images
  num_iterations = int(np.floor(num_images/config.batch_size))

  # create iterator allowing us to switch between datasets
  data_iterator = dataset.make_one_shot_iterator()
  next_element = data_iterator.get_next()

  # create placeholder for train or test
  train_network = tf.placeholder(tf.bool, [])

  # get images and pass into network
  image, label, weight = next_element
  drn = DRN(image, config.image_dims, config.batch_size, config.num_classes, 
            train_network, config.network)

  # get predictions and logits
  prediction = drn.pred
  logits = drn.prob
  label = tf.squeeze(label, axis=-1)

  # resize the logits using bilinear interpolation
  imsize = tf.constant([config.image_dims[0], config.image_dims[1]], dtype=tf.int32)
  logits = tf.image.resize_bilinear(logits, imsize)
  prediction = tf.argmax(logits, axis=-1)
  print('Resized shape is {}'.format(logits.get_shape()))

  # global step to keep track of iterations
  global_step = tf.Variable(0, trainable=False, name='global_step')

  saver = tf.train.Saver(max_to_keep=3)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
  
    # initialize variables 
    sess.run(init)
    
    # restore checkpiont if it exists
    ckpt = tf.train.get_checkpoint_state(config.logs)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path) 
      print('Restoring session at step {}'.format(global_step.eval()))
      
    iteration = global_step.eval()
    for i in range(num_iterations):
      print('step: {} of {}'.format(i, num_iterations))
      img, pred = sess.run([image, prediction], feed_dict={train_network:False})

      fnames = image_list[config.batch_size*i:config.batch_size*i + config.batch_size]
      # write images to file
      for j in range(pred.shape[0]):
        fname = fnames[j].split('/')[-1]

        # drop file extension
        fname = fname.split('.')[0]

        if not os.path.isdir(pred_directory):
          os.makedirs(pred_directory)

        img_write = im.fromarray(pred[j, :, :], "L")
        img_write.save(os.path.join(pred_directory, fname + ".png"))