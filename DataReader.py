import tensorflow as tf

class DataReader():
  def __init__(self, directory, batch_size, num_epochs, use_weights):
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.use_weights = use_weights
    self.directory = directory
    self.IMG_MEAN = tf.constant([60.3486, 60.3486, 60.3486], dtype=tf.float32)

  def read_files(self, data_file):
    f = open(data_file, 'r')
    data = f.read()
    data = data.split('\n')
    image_list = []
    label_list = []
    weight_list = []

    for i in range(len(data)):
      line = data[i]
      if line:
        image, label, weight = line.split(' ')
        image_list.append(self.directory + '/' + image)
        label_list.append(self.directory + '/' + label)
        weight_list.append(self.directory + '/' + weight)

    # store number of images and image filenames
    self.image_list = image_list
    self.label_list = label_list
    self.weight_list = weight_list
    self.num_images = len(data)
    
    return image_list, label_list, weight_list
    
  def decode_image(self, image, label, weight):
    # helper function to decode image
    image = tf.read_file(image)
    label = tf.read_file(label)
    weight = tf.read_file(weight)
    image = tf.image.decode_jpeg(image)
    label = tf.image.decode_png(label)
    weight = tf.image.decode_png(weight)
    
    # convert image to float
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)
    weight = tf.cast(weight, dtype=tf.float32)
    
    # subtract the training mean from the imagess
    image -= self.IMG_MEAN
    
    return image, label, weight

  def mirror_image(self, image, label, weight):
    # if passingsingle images in for random flipping
    cond = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
    label = tf.cond(cond, lambda: tf.image.flip_left_right(label), lambda: tf.identity(label))
    weight = tf.cond(cond, lambda: tf.image.flip_left_right(weight), lambda: tf.identity(weight))
    
    return image, label, weight
  
  def rotate_image(self, image, label, weight):
    # rotate images randomly
    rot_angle = tf.random_uniform([], minval=0, maxval=360, dtype=tf.float32)
    image = tf.contrib.image.rotate(image, rot_angle)
    label = tf.contrib.image.rotate(label, rot_angle)
    weight = tf.contrib.image.rotate(weight, rot_angle)
    
    
    return image, label, weight
   
  def add_noise(self, image, label, weight):
    # add gaussian noise
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1)
    image += noise
    
    # if uint8 format still
    #noise = tf.random_normal(shape=tf.shape(imag), mean=125, stddev=25)
    return image, label, weight
    
  def train_batch(self, train_file):
    image_list, label_list, weight_list = self.read_files(train_file)
    train_data = tf.data.Dataset.from_tensor_slices((image_list, label_list, weight_list))
    # shuffle all files
    train_data = train_data.shuffle(buffer_size=len(image_list))
    # decode images and subtract image mean
    train_data = train_data.map(self.decode_image)
    # Data augmentation
    train_data = train_data.map(self.rotate_image,  num_parallel_calls=2)
    train_data = train_data.map(self.mirror_image, num_parallel_calls=2)
    train_data = train_data.map(self.add_noise,  num_parallel_calls=2)
    train_data = train_data.repeat()
    #train_data = train_data.batch(self.batch_size)
    train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
 
    return train_data
    
  def test_batch(self, test_file):
    image_list, label_list, weight_list = self.read_files(test_file)
    test_data = tf.data.Dataset.from_tensor_slices((image_list, label_list, weight_list))
    test_data = test_data.map(self.decode_image)
    #test_data = test_data.batch(self.batch_size)
    test_data = test_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))

    return test_data