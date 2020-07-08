import tensorflow as tf

def dice_loss(logits, label, num_classes, use_weights):
  """Computes the DICE loss

  Args:
    logits: tensor, output logits/scores from neural network
    labels: tensor, ground truth labelmaps
    num_classes: int, number of classes
    use_weights: bool, Flag to weight class labels in loss
  Returns:
    dice_loss: int, loss evaluated as (1 - dice_coefficient)
  """

  num_classes = logits.shape[-1]

  logits = tf.nn.softmax(logits)

  label_one_hot = tf.one_hot(label, num_classes)

  # create weight for each class 
  w = tf.zeros((num_classes))
  w = tf.reduce_sum(label_one_hot, axis=[0,1,2])

  # optionally apply weights
  use_weights = tf.convert_to_tensor(use_weights)
  w = tf.cond(use_weights, lambda: 1/(w**2), lambda: tf.ones((num_classes)))

  # sum over batches and images
  ref_vol = tf.reduce_sum(label_one_hot, axis=[0,1,2]) + 0.1
  intersect = tf.reduce_sum(label_one_hot*logits, axis=[0,1,2])
  seg_vol = tf.reduce_sum(logits, [0,1,2]) + 0.1
  
  # sum over all classes
  dice_numerator = 2.0*tf.reduce_sum(tf.multiply(w, intersect))
  dice_denominator = tf.reduce_sum(tf.multiply(w, seg_vol + ref_vol))
 
  # subtract 1 as we are tyring to maximize the DICE but optimization will minimize
  dice_loss = 1.0 - dice_numerator / dice_denominator
 
  return dice_loss