import os
import tensorflow as tf
import numpy as np
from accuracy import jaccard, dice
from data_reader import DataReader
from drn import DRN
from dice import dice_loss

def train(config):
  """Trains the model based on configuration settings

  Args:
    config: configurations for training the model
  """

  tf.reset_default_graph()

  data = DataReader(config.directory, config.image_dims, config.batch_size, 
                    config.num_epochs, config.use_weights)
  train_data = data.train_batch(config.train_file)
  num_train_images = data.num_images

  test_data = data.test_batch(config.val_file)
  num_val_images = data.num_images

  # determine number of iterations based on number of images
  training_iterations = int(np.floor(num_train_images/config.batch_size))
  validation_iterations = int(np.floor(num_val_images/config.batch_size))

  # create iterators allowing us to switch between datasets
  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle, 
    train_data.output_types, train_data.output_shapes)
  next_element = iterator.get_next()
  training_iterator = train_data.make_initializable_iterator()
  val_iterator = test_data.make_initializable_iterator()

  # create placeholder for train or test
  train_network = tf.placeholder(tf.bool, [])

  # get images and pass into network
  image, label, weight = next_element
  drn = DRN(image, config.image_dims, config.batch_size, config.num_classes, 
            train_network, config.network)

  # get predictions and logits
  prediction = drn.pred
  logits = drn.prob
  label = tf.squeeze(label, 3)

  # resize the logits using bilinear interpolation
  imsize = tf.constant([config.image_dims[0], config.image_dims[1]], 
                        dtype=tf.int32)
  logits = tf.image.resize_bilinear(logits, imsize)
  print('Resized shape is {}'.format(logits.get_shape()))

  prediction = tf.argmax(logits, 3)

  if config.loss == 'CE':
    if config.use_weights:
      label_one_hot = tf.one_hot(label, config.num_classes)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, 
                                                      logits=logits)
      loss = loss*tf.squeeze(weight, 3)
    else:
      # use sparse with flattened labelmaps
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, 
                                                            logits=logits)
    loss = tf.reduce_mean(loss)
  elif config.loss == 'dice':
    loss = dice_loss(logits, label, config.num_classes, 
                      use_weights=config.use_weights)
  else:
    NameError("Loss must be specified as CE or DICE")

  # global step to keep track of iterations
  global_step = tf.Variable(0, trainable=False, name='global_step')

  # create placeholder for learning rate
  learning_rate = tf.placeholder(tf.float32, shape=[])

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

  saver = tf.train.Saver(max_to_keep=3)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(val_iterator.string_handle())
    
    sess.run(training_iterator.initializer)
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state(config.logs)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path) 
      print('Restoring session at step {}'.format(global_step.eval()))
      
    # if restoring saved checkpoint get last saved iteration so that correct
    # epoch can be restored
    iteration = global_step.eval()
    current_epoch = int(np.floor(iteration/training_iterations)) 
    
    while current_epoch < config.num_epochs:

      train_loss = 0
      for i in range(training_iterations):
        _, l = sess.run([optimizer, loss], feed_dict={handle:training_handle, 
          learning_rate:config.learning_rate, train_network:True})
        train_loss += l
        iteration = global_step.eval()

      sess.run(val_iterator.initializer)
      val_loss = 0
      for i in range(validation_iterations):
        l, img, lbl, pred = sess.run([loss, image, label, prediction], 
          feed_dict={handle:validation_handle, train_network:False})
        val_loss += l

        # evaluate accuracy
        accuracy = jaccard(lbl, pred, config.num_classes)
        dice_score = dice(lbl, pred, config.num_classes)

      print('Train loss Epoch {} step {} :{}'.format(current_epoch, iteration, 
        train_loss/training_iterations))
      print('Validation loss Epoch {} step {} :{}'.format(current_epoch, iteration, 
        val_loss/validation_iterations))

      with open('loss.txt', 'a') as f: 
        f.write("Epoch: {} Step: {} Loss: {}\n".format(current_epoch, iteration, 
          train_loss/training_iterations))

      saver.save(sess, config.logs + '/model.ckpt', global_step)

      current_epoch += 1 