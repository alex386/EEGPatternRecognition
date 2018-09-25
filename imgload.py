#
# Library for reading images and labels
#
import os
import glob
import cv2
from sklearn.utils import shuffle
import numpy as np


def load_train_set(train_path, width, height, img_labels):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Reading images ...')
    for fields in img_labels:   
        index = img_labels.index(fields)
        print('Class name {} (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            #iheight, iwidth, ichannels = image.shape
            image = cv2.resize(image, (width, height), 0, 0, cv2.INTER_LINEAR) # Resize image to the same with and height
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(img_labels))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class ImageSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      #print("numexamples ",self._num_examples)
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, set_size, image_w, image_h, classes):
  class ImageSets(object):
    pass
  image_sets = ImageSets()

  images, labels, img_names, cls = load_train_set(train_path, image_w, image_h, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls,random_state=0)

  print("img_names:\t{}".format(img_names))  

  if isinstance(set_size, float):
    set_size = int(set_size * images.shape[0])

  validation_labels = labels[:set_size]
  validation_images = images[:set_size]
  validation_cls = cls[:set_size]
  validation_img_names = img_names[:set_size]
  

  train_images = images[set_size:]
  train_labels = labels[set_size:]
  train_img_names = img_names[set_size:]
  train_cls = cls[set_size:]

  image_sets.train = ImageSet(train_images, train_labels, train_img_names, train_cls)
  image_sets.valid = ImageSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return image_sets


