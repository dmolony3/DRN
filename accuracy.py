from skimage import measure
import numpy as np

# get accuracy
def Jaccard(label, pred, num_classes):
  print((label.shape, pred.shape))
  IOU = np.zeros((label.shape[0], num_classes), dtype=np.float32)
  for i in range(num_classes):
    inter = np.multiply(label == i, pred == i)
    inter = np.sum(inter, axis=(1,2))
    union = np.subtract(np.add(np.sum(label == i, axis=(1,2)), np.sum(pred == i, axis=(1,2))), inter)
    IOU[:, i] = inter/union
  IOU = np.mean(IOU, 0)
    
  return IOU

def DICE(label, pred, num_classes):
  dice = np.zeros((label.shape[0], num_classes), dtype=np.float32)
  for i in range(num_classes):
    inter = np.multiply(label == i, pred == i)
    inter = np.sum(inter, axis=(1,2))
    union = np.add(np.sum(label == i, axis=(1,2)), np.sum(pred == i, axis=(1,2)))
    dice[:, i] = 2*inter/union
  dice = np.mean(dice, 0)
    
  return dice
  
def Hausdorff(label, pred, num_classes):
  
  hauss = np.zeros((label.shape[0], num_classes), dtype=np.float32)
  
  levels = np.arange(0, num_classes) + 0.5
  # iterate over each contour
  for i, level in enumerate(levels):
    # iterate over each image in the batch
    for j in range(label.shape[0]):
      label_contour = measure.find_contours(label[j, :, :], level)
      pred_contour = measure.find_contours(pred[j, :, :], level)
      
      # convert label and pred to contours where rows are observations and columns are dimensions
      P = np.asarray(label_contour[0])
      Q = np.asarray(pred_contour[0])
      
      lenP = P.shape[0]
      lenQ = Q.shape[0]
      
      D = np.zeros((lenP, lenQ))
      
      for ii in range(0, lenP):
        for jj in range(0, lenQ):
          D[ii- 1, jj - 1] = np.sqrt((P[ii,0] - Q[jj, 0])**2 + (P[ii, 1] - Q[jj, 1])**2)
          
      d1 = np.max(np.min(D, axis=1))
      d2 = np.max(np.min(D, axis=0))
      hauss[j, i] = np.maximum(d1, d2)
      
  return hauss
    