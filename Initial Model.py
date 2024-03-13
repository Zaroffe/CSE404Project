import os

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_hub as tf_hub

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


my_seed = 42
split_proportion = 0.1
batch_size = 4
number_epochs = 8
lr = 3e-5

figure_count = 0
figure_dir = os.path.join("..", "assets")
if os.path.exists(figure_dir):
    pass
else:
    os.mkdir(figure_dir)

train_new = True




x = np.load("X.npy")
y = np.load("Y.npy")


# Count instances per class
unique, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(10, 6))
plt.bar(unique, counts)
plt.xticks(unique, rotation=65)
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.show()


# shuffle and split the data
split_number = int(split_proportion * x.shape[0])

np.random.seed(my_seed)
np.random.shuffle(x)
val_x = tf.convert_to_tensor(x[:split_number])
test_x = tf.convert_to_tensor(x[split_number:2*split_number])
train_x = tf.convert_to_tensor(x[2*split_number:])

np.random.seed(my_seed)
np.random.shuffle(y)
val_y_labels = tf.convert_to_tensor(y[:split_number])
test_y_labels = tf.convert_to_tensor(y[split_number:2*split_number])
train_y_labels = tf.convert_to_tensor(y[2*split_number:])

# visualize images with labels

fig, ax = plt.subplots(3,3, figsize=(8,8))
for count, x_index in enumerate(np.random.randint(0, train_x.shape[0], size=(9,))):

    cx = count // 3
    cy = count % 3
    ax[cx,cy].imshow(train_x[x_index])
    ax[cx,cy].set_title(f"label: {train_y_labels[x_index]}")
    ax[cx,cy].set_yticklabels("")
    ax[cx,cy].set_xticklabels("")
    
plt.savefig(os.path.join(figure_dir, "figure_{figure_count}.png"))
figure_count += 1
plt.tight_layout()
plt.show()


label_dict = {}

for number, label in enumerate(np.unique(train_y_labels)):
    label_dict[number] = label
    
print(label_dict, x.shape)

reverse_label_dict = {}
for key in label_dict.keys():
    reverse_label_dict[label_dict[key]] = key
    
print(reverse_label_dict)

np_train_y = np.zeros_like(train_y_labels) # , dtype=tf.int32)
np_val_y = np.zeros_like(val_y_labels)
np_test_y = np.zeros_like(test_y_labels)

for ii in range(np_train_y.shape[0]):
    np_train_y[ii] = reverse_label_dict[train_y_labels[ii].numpy()[0]]
    
for ii in range(np_val_y.shape[0]):
    np_val_y[ii] = reverse_label_dict[val_y_labels[ii].numpy()[0]]
    
for ii in range(np_test_y.shape[0]):
    np_test_y[ii] = reverse_label_dict[test_y_labels[ii].numpy()[0]]
    
train_y = tf.convert_to_tensor(np_train_y.reshape(-1), dtype=tf.int32)
val_y = tf.convert_to_tensor(np_val_y.reshape(-1), dtype=tf.int32)
test_y = tf.convert_to_tensor(np_test_y.reshape(-1), dtype=tf.int32)

# visualize images with labels

fig, ax = plt.subplots(3,3, figsize=(8,8))
for count, x_index in enumerate(np.random.randint(0, val_x.shape[0], size=(9,))):

    cx = count // 3
    cy = count % 3
    idx = val_y[x_index]
    ax[cx,cy].imshow(val_x[x_index])
    ax[cx,cy].set_title(f"label index: \n {idx} = {label_dict[idx.numpy()]}")
    ax[cx,cy].set_yticklabels("")
    ax[cx,cy].set_xticklabels("")
    
plt.tight_layout()

plt.savefig(os.path.join(figure_dir, "figure_{figure_count}.png"))
figure_count += 1

plt.show()



number_classes = len(label_dict.keys())

extractor = tf.keras.applications.MobileNet(\
    input_shape=train_x.shape[1:], include_top=False,weights="imagenet")
    

extractor.trainable = True


model = Sequential([extractor, \
        tf.keras.layers.Flatten(),\
        tf.keras.layers.Dropout(0.25),\
        Dense(32, activation="relu"),\
        Dense(32, activation="relu"),\
        Dense(number_classes, activation="softmax")])

#model.build([None, 128, 128, 3])


_ = model(train_x[0:1])
model.summary()

model.compile(optimizer = 'adam',\
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy']
             )


model.summary()


correct_keras = 0
total_samples = test_x.shape[0]

for my_index in range(test_x.shape[0]):

    my_batch = test_x[my_index:my_index+1]
    
    full_output_data = model(my_batch)
                        
    
    true_label = test_y[my_index].numpy()
    
    correct_keras += 1.0 * (full_output_data.numpy().argmax() == true_label)
    

accuracy_keras = correct_keras / total_samples

msg = f"Test accuracies "
msg += f"\n\t keras {accuracy_keras:.4f}"

print(msg)
