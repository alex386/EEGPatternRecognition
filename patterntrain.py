# Tensorflow EEG Pattern training
# Author: Aleksander Dawid

import imgload
import tensorflow as tf
import time

#All random initializations uses the same seed=0
from numpy.random import seed
seed(0)

from tensorflow import set_random_seed
set_random_seed(0)

old_val_loss=1000.0

#Prepare input data
classes = ['LIP','JAW','NONE']
num_classes = len(classes)

# 20% of the input data will be used for validation
validation_size = 0.2
img_size_w = 256 #image width
img_size_h = 64 #image height
num_channels = 3
# Set path where images are located in classes directories
train_path='TEST' 
pathSession = train_path+"/Session"
pathLog = pathSession+"/Learn.log"
pathModel=train_path+"/Model/"+train_path+"Model"

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = imgload.read_train_sets(train_path, validation_size, img_size_w, img_size_h, classes)

notrain=len(data.train.labels)
novalid=len(data.valid.labels)
nototal=notrain+novalid

print("Image dataset info")
print("Total Images:\t{}".format(nototal))
print("Training-set:\t{}".format(notrain))
print("Validation-set:\t{}".format(novalid))

#data.__format__
batch_size = len(data.valid.labels)
#print("Batch size:\t{}".format(batch_size))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Enable GPU calculations

session = tf.Session(config=config)

x = tf.placeholder(tf.float32, shape=[None, img_size_h,img_size_w,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


##Network graph params
filter_size_conv1 = 5 
num_filters_conv1 = 32

filter_size_conv2 = 5
num_filters_conv2 = 32

filter_size_conv3 = 5
num_filters_conv3 = 64

filter_size_conv4 = 4
num_filters_conv4 = 64

lrate=2e-5
    
fc_layer_size = img_size_w
fc1_layer_size = 256
fc2_layer_size = 256

#fc_layer_size = 256

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, seed=0))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

#Convolution layers
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

layer_conv4 = create_convolutional_layer(input=layer_conv3,
               num_input_channels=num_filters_conv3,
               conv_filter_size=filter_size_conv4,
               num_filters=num_filters_conv4)

      
layer_flat = create_flatten_layer(layer_conv4)


#Number of elements in flat layer
fc1_elements=layer_flat.get_shape()[1:4].num_elements()

print("Number of neurons FC1:\t{}".format(fc1_elements))
print("Number of neurons FC2:\t{}".format(fc1_layer_size))


# Fully connected layers
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc1_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc1_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer(), options=run_options)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
castvalue=tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(castvalue)


session.run(tf.global_variables_initializer(), options=run_options) 




def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    global lrate,fc1_elements,iiter
    
    
    if epoch%10==0 and epoch>9:
        lrate=lrate/5

    
    acc = session.run(accuracy, feed_dict=feed_dict_train, options=run_options)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate, options=run_options)
    msg = "Epoch {0} --- Training: {1:>6.1%}, Validation: {2:>6.1%},  VLoss: {3:.3f} iter: {4}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss,iiter))
    
    msg2 = "{0} {1:.6f} {2:.6f} {3:.6f} \n" 
    
    file = open(pathLog,"a"); 
    file.write(msg2.format(epoch + 1, acc, val_acc, val_loss))
    file.close();
    

total_iterations = 0
iiter=0



saver = tf.train.Saver()
writer = tf.summary.FileWriter(pathSession, session.graph)

def train(num_iteration):
    global total_iterations
    global old_val_loss,lrate,iiter
    
    time1 = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iteration):
        iiter=iiter+1
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr, options=run_options)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val, options=run_options)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            if old_val_loss-val_loss<0:
                break
            old_val_loss=val_loss
    time2 = time.time()
    saver.save(session, pathModel)
    total_iterations += num_iteration
    print("Total time=",time2-time1," [s]")
train(num_iteration=400)


