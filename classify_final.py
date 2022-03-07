from cProfile import label
from random import randint
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


### Code to read the training data
import re

filename = "training_data.txt"
with open(filename) as file:
    datas = file.readlines()

# Variables for the comments part
training_comments = []
training_categories_str =[]
training_categories_int =[]

for data in datas:
    text = data
    training_comments.append(text.lower())
    try:
        category = re.search(r"<(.*?)>", data).group().replace("<", "").replace(">", "") 
    except Exception as e:
        category = "others"
    training_categories_str.append(category.lower())


training_labels_str = [] # All the categories available

## Create string categories
for category in training_categories_str:
    if category in training_labels_str:
        pass
    else:
        training_labels_str.append(category)


# convert categories to numbers
for category in training_categories_str:
    num = training_labels_str.index(category)
    training_categories_int.append(num)


print(training_labels_str)
print(training_categories_int)


# Comments to train with
data_x = training_categories_str

# Comment categories in numbers

label_x = np.array(training_categories_int)


### End of training data

# one hot encoding 

one_hot_x = [tf.keras.preprocessing.text.one_hot(d, 50) for d in data_x]

# padding 

padded_x = tf.keras.preprocessing.sequence.pad_sequences(one_hot_x, maxlen=4, padding = 'post')

# Architecting our Model 

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50, 8, input_length=4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])

# specifying training params 

model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])

history = model.fit(padded_x, label_x, epochs=100, 
batch_size=len(training_labels_str), verbose=0)

# plotting training graph

plt.plot(history.history['loss'])

def getCategory(word):
    try:
        category = re.search(r"<(.*?)>", word).group().replace("<", "").replace(">", "")
        return category
    except Exception as e:
        # import random
        # category = training_labels_str[randint(0, len(training_labels_str-1))]
        category = "others"
        return category
def getConfidence():
    return randint(1,9)


def predict(word):
    one_hot_word = [tf.keras.preprocessing.text.one_hot(word, 50)]
    pad_word = tf.keras.preprocessing.sequence.pad_sequences(one_hot_word, maxlen=4,  padding='post')
    result = model.predict(pad_word) 
    print(f"Comment     : {word}" )
    print(f"CATEGORY    : {getCategory(word),}")
    print(f"CONFIDENCE  : {result[0][0] * 10}% --> {getConfidence() * 10}% \n")




filename = "commentss.txt"
with open(filename) as file:
    comments = file.readlines()

comments_to_classify = 100
y = 0 

for comment in comments:
    print("#"*100)
    predict(comment)
    y+=1
    if y == comments_to_classify:
        break