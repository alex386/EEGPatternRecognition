# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:55:47 2018

@name: CSVMachLearn.py
@description: 1D CNN using CSV vector for machine learning
@author: Aleksander Dawid
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from sklearn.decomposition import PCA

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow import set_random_seed

tf.enable_eager_execution()
set_random_seed(0)

nrds='S0'
#==============================================================================
# Global parameters
#==============================================================================
total_dataset_fp="D:\\AI_experiments\\CSV\\"+nrds+"\\DAT"+nrds+".csv"
pathlog="D:\\AI_experiments\\CSV\\"+nrds+"\\"+nrds+"pub.log"
pathimg="D:\\AI_experiments\\CSV\\"+nrds+"\\IMG"
num_epochs = 1001                                      # number of epochs
lrate=2e-5                                            # learning rate
test_procent=0.2                                      # procentage of test_dataset
learn_batch_size=32                                   # batch size

print("Local copy of the dataset file: {}".format(total_dataset_fp))



print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#==============================================================================
# Methods
#==============================================================================
def ChangeBatchSize(dataset,bsize):
    dataset=dataset.apply(tf.data.experimental.unbatch())
    dataset=dataset.batch(batch_size=bsize)
    return dataset

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

with open(total_dataset_fp) as f:
    content = f.readlines()
grup=content[0].split(',')
print(grup[1])

f_size=int(grup[1])-1   #number of points in data vector 
print("Vector size: "+str(f_size))


filtr1=32
filtr_size1=5

filtr2=32
filtr_size2=5

filtr3=64
filtr_size3=5

filtr4=64
filtr_size4=4

DenseLast=4096

filtr5=512
filtr_size5=5

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((f_size,1), input_shape=(None,f_size),name='x'),
    tf.keras.layers.Conv1D(filters=filtr1,kernel_size=filtr_size1,strides=1, kernel_initializer='random_uniform',activation=tf.nn.relu,padding='same',name='Conv1'),
    tf.keras.layers.MaxPooling1D(pool_size=filtr_size1, strides=2, padding='same', name='pool1'),
    tf.keras.layers.Conv1D(filters=filtr2,kernel_size=filtr_size2,strides=1, padding='same',name='Conv2',activation=tf.nn.relu, kernel_initializer='random_uniform'),
    tf.keras.layers.MaxPooling1D(pool_size=filtr_size2, strides=2, padding='same', name='pool2'),
    
    tf.keras.layers.Conv1D(filters=filtr3,kernel_size=filtr_size3,strides=1, padding='same',name='Conv3',activation=tf.nn.relu, kernel_initializer='random_uniform'),
    tf.keras.layers.MaxPooling1D(pool_size=filtr_size3, strides=2, padding='same', name='pool3'),
    tf.keras.layers.Conv1D(filters=filtr4,kernel_size=filtr_size4,strides=1, padding='same',name='Conv4',activation=tf.nn.relu, kernel_initializer='random_uniform'),
    tf.keras.layers.MaxPooling1D(pool_size=filtr_size4, strides=2, padding='same', name='pool4'),

    tf.keras.layers.GlobalMaxPool1D(),    #size of last filter

    tf.keras.layers.Dense(DenseLast, activation=tf.nn.relu,name='fir'),  # input shape required
    tf.keras.layers.Dense(256, activation=tf.nn.relu,name='mod_up'),

    tf.keras.layers.Dense(3,name='y_pred'), #output layer

  ])
  
  model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

def loss(model, x, y):
  y_ = model(x)
  #print(y)
  #print(y_)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
    #print(loss_value)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


mapcolor=['red','green','blue']

# column order in CSV file
column_names = []

for a in range(0,f_size):
    column_names.append(str(a))

column_names.append('signal')

print(len(column_names))

feature_names = column_names[:-1]
label_name = column_names[-1]

#class_names = ['Left','Right','NONE']
class_names = ['LIP','JAW','NONE']


batch_size = 200000

#train_dataset = tf.data.experimental.make_csv_dataset(
#    total_dataset_fp,
#    batch_size, 
#    column_names=column_names,
#    label_name=label_name,
#    num_epochs=1,
#    shuffle=False)

#train_dataset = train_dataset.map(pack_features_vector)

total_dataset = tf.data.experimental.make_csv_dataset(
    total_dataset_fp,
    batch_size, 
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
    shuffle=True)

features, labels = next(iter(total_dataset))
setsize=float(str(labels.shape[0]))
ts_size=setsize*test_procent
tr_size=setsize-ts_size
print("Total_CSV_size: "+str(setsize) )
print("Train_size: "+str(tr_size) )
print("Test_size: "+str(ts_size) )


total_dataset = total_dataset.map(pack_features_vector)
total_dataset=ChangeBatchSize(total_dataset,tr_size)

#==============================================================================
#Split dataset into train_dataset and test_dataset.
#==============================================================================
i=0
for (parts, labels) in total_dataset:
    if(i==0):
       k1 = parts
       l1 = labels
    else:
       k2 = parts
       l2 = labels
    i=i+1

train_dataset = tf.data.Dataset.from_tensors((k1, l1))
train_dataset = ChangeBatchSize(train_dataset,learn_batch_size) 

test_dataset = tf.data.Dataset.from_tensors((k2, l2))
test_dataset = ChangeBatchSize(test_dataset,ts_size) 

#==============================================================================
# Create model object
#==============================================================================

model=create_model()
model.summary()

optimizer = tf.train.AdamOptimizer(learning_rate=lrate)

global_step = tf.train.get_or_create_global_step()


legend_elements = [Line2D([0], [0], marker='o', color='w', label=class_names[0],markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=class_names[1],markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=class_names[2],markerfacecolor='b', markersize=10)]
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

np.set_printoptions(threshold=np.nan)
#==============================================================================
# Make machine learning process
#==============================================================================
old_loss=1000

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    #print(str(type(x)))
    #print(str(x.shape))
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  

 
  if epoch % 5 == 0:
    test_accuracy = tfe.metrics.Accuracy()

    for (x, y) in test_dataset:
      logits = model(x)
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
      test_accuracy(prediction, y)
      X=logits.numpy()
      Y=y.numpy()
      PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
      X = PCA(n_components=2).fit_transform(X)

      arrcolor = []

      for cl in Y:  
        arrcolor.append(mapcolor[cl])
        
      plt.scatter(X[:, 0], X[:, 1], s=40, c=arrcolor)
      #plt.show()
      imgfile="{:s}\\epoch{:03d}.png".format(pathimg,epoch)
      plt.title("{:.3%}".format(test_accuracy.result()))                  
      plt.legend(handles=legend_elements, loc='upper right')
      plt.savefig(imgfile)
      plt.close()
    
    new_loss=epoch_loss_avg.result()
    accur=epoch_accuracy.result()
    test_acc=test_accuracy.result()
    msg="Epoch {:03d}: Loss: {:.6f}, Accuracy: {:.3%}, Test: {:.3%}".format(epoch,new_loss,accur,test_acc)
    msg2 = "{0} {1:.6f} {2:.6f} {3:.6f} \n".format(epoch,accur,test_acc,new_loss)
    print(msg)
    
    
    
    if new_loss>old_loss:
        break
    file = open(pathlog,"a"); 
    file.write(msg2)
    file.close();
    old_loss=epoch_loss_avg.result()
    
#==============================================================================
# Save trained model to disk
#==============================================================================

model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

filepath="csvsignal.h5"
tf.keras.models.save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True
)

print("Model csvsignal.h5 saved to disk")

