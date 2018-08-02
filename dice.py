import tensorflow as tf

def dice(logits, label, num_classes, use_weights):
  # logits must by 4d tensor, label should be 3d tensor
  # https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/169d31b62d9588e5acff7effdf93fb1f2c3c22f7/niftynet/layer/loss.py
  num_classes = logits.shape[-1]
  logits = tf.nn.softmax(logits)
  label_one_hot = tf.one_hot(label, num_classes)

  # to generate weights
  if use_weights == 1:
    w = tf.zeros((num_classes))
    # create weight for each class 
    w = tf.reduce_sum(label_one_hot, axis=[0,1,2])
    w = 1/(w**2)
  else:
    w = 1
   # sum over batches and images
  ref_vol = tf.reduce_sum(label_one_hot, axis=[0,1,2]) + 0.1
  intersect = tf.reduce_sum(label_one_hot * logits, axis=[0,1,2])
  seg_vol = tf.reduce_sum(logits, [0,1,2]) + 0.1
  
  # sum over all classes
  dice_numerator = 2.0 * tf.reduce_sum(tf.multiply(w, intersect))
  dice_denominator = tf.reduce_sum(tf.multiply(w, seg_vol + ref_vol))
 
  # subtract 1 as we are tyring to maximize the DICE but optimization will minimize
  dice_loss = 1.0 - dice_numerator / dice_denominator
 
  return dice_loss