""" Transfer Learning Module DOC STRING

"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image_dataset_from_directory

def predict(model, input_batch, labels=None):

  predictions = model.predict_on_batch(input_batch).flatten()

  # Apply a sigmoid since our model returns logits
  predictions = tf.nn.sigmoid(predictions)
  predictions = tf.where(predictions < 0.5, 0, 1)

  print('Predictions:\n', predictions.numpy())
  if labels is not None:
    print('Labels:\n', labels)

  # plt.figure(figsize=(10, 10))
  # for i in range(len(predictions)):
  #   ax = plt.subplot(3, 3, i + 1)
  #   plt.imshow(image_batch[i].astype("uint8"))
  #   plt.title(class_names[predictions[i]])
  #   plt.axis("off")


if __name__ == "__main__":

  BATCH_SIZE = 32
  IMG_SIZE = (512, 512)

  dataset_path = os.path.join(
      "..","data","datasets","datasets","ALDataset"
      )

  img_path = os.path.join(dataset_path, "Leak", "image_02.png")
  image = np.asarray(Image.open(img_path))
  # train_dataset = image_dataset_from_directory(
  #     os.path.join(dataset_path, "train"),
  #     shuffle=True,
  #     batch_size=BATCH_SIZE,
  #     image_size=IMG_SIZE
  #   )

  # test_dataset = image_dataset_from_directory(
  #     os.path.join(dataset_path, "test"),
  #     shuffle=True,
  #     batch_size=BATCH_SIZE,
  #     image_size=IMG_SIZE
  #   )

  #train_dataset = train_dataset.prefetch(4)
  # test_dataset = test_dataset.prefetch(4)

  # base_model_uri = "https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1"
  base_model_uri = "https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/2"

  base_model = hub.KerasLayer(base_model_uri, trainable=False)

  #base_model.trainable = False

  inputs = tf.keras.Input(shape=(640,640,3))
  x = base_model(inputs) #base_model(inputs, training=False)
  model= tf.keras.Model(inputs, x)

  # base_learning_rate = 0.0001
  # model.compile(optimizer= \
  #               tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
  #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #               metrics=['accuracy'])

  print(model.summary())
















  # print(image.shape)

  # pred = base_model.predict(image[np.newaxis,...])

  # print(pred)

  #------------------------- 
  # Apply image detector on a single image.
  #detector = hub.load(base_model_uri)


  #global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  ## feature_batch_average = global_average_layer(feature_batch)

  #prediction_layer = tf.keras.layers.Dense(1)
  ## prediction_batch = prediction_layer(feature_batch_average)

  #inputs = tf.keras.Input(shape=(160,160,3))
  #x = data_augmentation(inputs)
  #x = preprocess_input(x)
  #x = base_model(x, training=False)
  #x = global_average_layer(x)
  #x = tf.keras.layers.Dropout(0.2)(x)
  #outputs = prediction_layer(x)
  #model = tf.keras.Model(inputs, outputs)

  #base_learning_rate = 0.0001
  #model.compile(optimizer= \
  #              tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
  #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #              metrics=['accuracy'])















  #print(model.summary())














  # detector_output = detector(image[np.newaxis,...])

  # x = detector_output['detection_boxes'].numpy().astype(np.int32) #100 x 4
  # y = detector_output['detection_scores'].numpy()
  # z = detector_output['num_detections'].numpy().astype(np.int32)[0]

  #image_batch, label_batch = next(iter(train_dataset))
  ## feature_batch = base_model(image_batch)

  #base_model.trainable = False

  #global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  ## feature_batch_average = global_average_layer(feature_batch)

  #prediction_layer = tf.keras.layers.Dense(1)
  ## prediction_batch = prediction_layer(feature_batch_average)

  #inputs = tf.keras.Input(shape=(160,160,3))
  #x = data_augmentation(inputs)
  #x = preprocess_input(x)
  #x = base_model(x, training=False)
  #x = global_average_layer(x)
  #x = tf.keras.layers.Dropout(0.2)(x)
  #outputs = prediction_layer(x)
  #model = tf.keras.Model(inputs, outputs)

  #base_learning_rate = 0.0001
  #model.compile(optimizer= \
  #              tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
  #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #              metrics=['accuracy'])

  #print(model.summary())

  #initial_epochs = 10
  #loss0, accuracy0 = model.evaluate(test_dataset)

  #print("initial loss: {:.2f}".format(loss0))
  #print("initial accuracy: {:.2f}".format(accuracy0))

  #history = model.fit(train_dataset,
  #                    epochs=initial_epochs,
  #                    validation_data=test_dataset)

  #model.save("no_fine_chkpt/")


  ###################################################################
  ## acc = history.history['accuracy']
  ## val_acc = history.history['val_accuracy']
  
  ## loss = history.history['loss']
  ## val_loss = history.history['val_loss']
  
  ## plt.figure(figsize=(8, 8))
  ## plt.subplot(2, 1, 1)
  ## plt.plot(acc, label='Training Accuracy')
  ## plt.plot(val_acc, label='Validation Accuracy')
  ## plt.legend(loc='lower right')
  ## plt.ylabel('Accuracy')
  ## plt.ylim([min(plt.ylim()),1])
  ## plt.title('Training and Validation Accuracy')
  
  ## plt.subplot(2, 1, 2)
  ## plt.plot(loss, label='Training Loss')
  ## plt.plot(val_loss, label='Validation Loss')
  ## plt.legend(loc='upper right')
  ## plt.ylabel('Cross Entropy')
  ## plt.ylim([0,1.0])
  ## plt.title('Training and Validation Loss')
  ## plt.xlabel('epoch')
  ## plt.show()

  ###################################################################

  #fine_tune_at = 100

  #base_model.trainable = True

  #for layer in base_model.layers[:fine_tune_at]:
  #  layer.trainable = False


  #model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #              optimizer = tf.keras.optimizers.RMSprop(
  #                  learning_rate=base_learning_rate / 10),
  #              metrics=['accuracy'])

  #print(model.summary())

  #fine_tune_epochs = 10
  #total_epochs = initial_epochs + fine_tune_epochs

  #history_fine = model.fit(train_dataset,
  #                         epochs=total_epochs,
  #                         initial_epoch=history.epoch[-1],
  #                         validation_data=test_dataset)

  #model.save("fine_chkpt/")

  #loss, accuracy = model.evaluate(test_dataset)
  #print('loss = {},\naccuracy = {}'.format(loss, accuracy))

