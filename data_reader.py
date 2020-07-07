import tensorflow as tf
import os

class DataReader():
  """Class for loading and batching images, labelmaps and weightmaps.

  Args:
    directory: string, path to directory containing images
    batch_size: int, number of samples in batch
    num_epochs: int, number of train/test epochs
    use_weights: bool, flag indicating whether weightmaps are included
  """
  
  def __init__(self, directory, image_size, batch_size, num_epochs, use_weights):
    self.image_size = image_size
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.use_weights = use_weights
    self.directory = directory
    self.image_list = []
    self.IMG_MEAN = tf.constant([60.3486, 60.3486, 60.3486], dtype=tf.float32)

  def read_files(self, data_file):
    """Reads files and returns list of images, labels (and weights)

    Args:
      data_file: string, path to file containing rows of image/label paths
    Returns:
      image_list: list, full path to each image
      label_list: list, full path to each labelmap
      weight_list: list, full path to each weightmap
    """

    f = open(data_file, 'r')
    data = f.read()
    data = data.split('\n')
    image_list = []
    label_list = []
    weight_list = []

    for i in range(len(data)):
      line = data[i]
      if line:
        try:
          image, label, weight = line.split(' ')
          image_list.append(os.path.join(self.directory,  image))
          label_list.append(os.path.join(self.directory, label))
          weight_list.append(os.path.join(self.directory, weight))
        except ValueError:
          try:
            image, label = line.split(' ')
            image_list.append(os.path.join(self.directory,  image))
            label_list.append(os.path.join(self.directory, label))
            weight_list.append('')
          except ValueError:
            image = line
            image_list.append(os.path.join(self.directory,  image))
            label_list.append('')
            weight_list.append('')

    self.num_images = len(data)
    
    return image_list, label_list, weight_list
    
  def decode_image(self, image_path, label_path, weight_path):
    """Reads image, label and weight paths and decodes
    
    Args:
      image_path: string, path to image 
      label_path: string, path to labelmap
      weight_path: string, path to weightmap
    Returns:
      image: 3D tensor, single image
      label: 2D tensor, single labelmap
      weight: 2D tensor, single weightmap
    """

    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32)

    """
    if label_path:
      label = tf.read_file(label_path)
      label = tf.image.decode_png(label)
      label = tf.cast(label, dtype=tf.int32)
    else:
      label = tf.zeros((tf.shape(image)[0], tf.shape(image)[1]))

    if weight_path:
      weight = tf.read_file(weight_path)
      weight = tf.image.decode_png(weight)
      weight = tf.cast(weight, dtype=tf.float32)
    else:
      weight = tf.ones((tf.shape(image)[0], tf.shape(image)[1]))
    """

    label = tf.cond(tf.cast(tf.strings.length(label_path), dtype=tf.bool), 
                    lambda: self.read_and_decode_png(label_path), 
                    lambda: tf.zeros(self.image_size, dtype=tf.uint8))

    weight = tf.cond(tf.cast(tf.strings.length(weight_path), dtype=tf.bool), 
                      lambda: self.read_and_decode_png(weight_path),
                      lambda: tf.zeros(self.image_size, dtype=tf.uint8))

    label = tf.cast(label, dtype=tf.int32)
    weight = tf.cast(weight, dtype=tf.float32)
    image -= self.IMG_MEAN

    return image, label, weight

  def read_and_decode_png(self, file_path):
    """Reads and decodes png files"""

    image = tf.read_file(file_path)
    image = tf.image.decode_png(image)    
    
    return image

  def mirror_image(self, image, label, weight):
    """Performs random flipping of image/labelmap/weightmap"""

    cond = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
    label = tf.cond(cond, lambda: tf.image.flip_left_right(label), lambda: tf.identity(label))
    weight = tf.cond(cond, lambda: tf.image.flip_left_right(weight), lambda: tf.identity(weight))
    
    return image, label, weight
  
  def rotate_image(self, image, label, weight):
    """Performs random rotation of image/labelmap/weightmap"""

    rot_angle = tf.random_uniform([], minval=0, maxval=360, dtype=tf.float32)
    image = tf.contrib.image.rotate(image, rot_angle)
    label = tf.contrib.image.rotate(label, rot_angle)
    weight = tf.contrib.image.rotate(weight, rot_angle)
    
    return image, label, weight
   
  def add_noise(self, image, label, weight):
    """Adds gaussian noise to input image"""

    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1)
    image += noise
    
    return image, label, weight
    
  def train_batch(self, train_file):
    """Reads and batches images for training

    Args:
      train_file: string, path to file containing rows of image/label paths
    Returns:
      train_data: tensorflow dataset, augmented batch of images/labels/weights
    """

    image_list, label_list, weight_list = self.read_files(train_file)
    self.image_list = image_list

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

    train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
 
    return train_data
    
  def test_batch(self, test_file):
    """Reads and batches images for testing

    Args:
      test_file: string, path to file containing rows of image/label paths
    Returns:
      test_data: tensorflow dataset, batch of images/labels/weights
    """

    image_list, label_list, weight_list = self.read_files(test_file)
    self.image_list = image_list

    test_data = tf.data.Dataset.from_tensor_slices((image_list, label_list, weight_list))
    test_data = test_data.map(self.decode_image)

    test_data = test_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))

    return test_data