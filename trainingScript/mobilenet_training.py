"""
This script is meant to train and evaluate the possibility of MobileNet as the classifier.
MobileNet is meant to by used for edge devices, and is available in Keras.
"""

print("######################################################################")
print("############################### Import ###############################")
print("######################################################################")

print("Imports os...")
import os

print("Imports MatplotLib...")
import matplotlib.pyplot as plt

print("Imports Numpy...")
import numpy as np

print("Imports TensorFlow...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silences TensorFlow generic warnings
import tensorflow as tf

print("Imports Keras...")
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

print("########################### Import - Done ############################\n")


print("######################################################################")
print("############################ Data loading ############################")
print("######################################################################")
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
print(f"Batch size was initialized to {BATCH_SIZE}")
print(f"Image size was initialized to {IMG_SIZE}\n")

directory = os.path.abspath("../data/SignLanguageNumbers/")
print(f"Loads data from {directory}\n")

"""
image_dataset_from_directory is one of the preprocessing functions that Keras provides
it takes the directory path of the data(which is organized in sub-directories named on
the gold labels of the images).
"""
train_dataset, validation_dataset = image_dataset_from_directory(directory,
                                                                 label_mode='categorical',
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 validation_split=0.2,
                                                                 subset='both',
                                                                 seed=42)

# Take a look of some of the images
class_names = train_dataset.class_names

print("\nLet's take a look on some of the samples in the training set")
plt.figure(figsize=(10, 10))
plt.title("Training samples")
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
plt.show()

print("######################## Data loading - Done #########################\n")


print("######################################################################")
print("########################### Data augmenter ###########################")
print("######################################################################")
# Creating a data augmenter
def data_augmenter():
    """
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    """
    data_augmentation = tf.keras.Sequential([])
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation

data_augmentation = data_augmenter()
print("A data augmenter was successfully created")

print("\nLet's take a look of some augmentation examples")
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
plt.show()

print("####################### Data augmenter - Done ########################\n")


print("######################################################################")
print("####################### MobileNet-V2 creation ########################")
print("######################################################################")
# Pre-processing algorithm provided by Keras
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# The MobileNet-V2 creation algorithm
def create_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
        tf.keras.model
    '''
    # We will use RGB images
    input_shape = image_shape + (3,)

    # We create our base model from the Keras's MobileNet-V2 implementation.
    # We removed the top of the model and change it for our needs(10 classification labels compared to 1000 in the
    # pre-trained original model)
    # We will use the "transfer learning" method, based on the ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    # freeze the base model by making it non-trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNet-V2 input size)
    inputs = tf.keras.Input(shape=input_shape, batch_size=1)

    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # apply the base model on the input
    x = base_model(x, training=False)

    # add the new 10-label classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)

    # use a prediction layer with 10 neurons
    outputs = tfl.Dense(10)(x)

    # create the final instance of the complete model
    model = tf.keras.Model(inputs, outputs)

    return model


# create our own instance of the model
model = create_model(IMG_SIZE, data_augmentation)
print("\nThe model was built successfully!")
print(model.summary())

print("#################### MobileNet-V2 creation - Done ####################\n")

print("######################################################################")
print("####################### MobileNet-V2 training ########################")
print("######################################################################")
base_learning_rate = 0.001
print(f"Chose learning rate: {base_learning_rate}")
print("Chosen optimizer: Adam\n")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model compilation was done!")
initial_epochs = 5
print(f"Number of epochs: {initial_epochs}")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

print("#################### MobileNet-V2 training - Done ####################\n")

print("######################################################################")
print("########################## Training results ##########################")
print("######################################################################")
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Graph printing
tf.keras.utils.plot_model(model, show_shapes=True, to_file="total_model.png", expand_nested=True)

print("####################### Training results - Done ######################\n")

print("######################################################################")
print("############################ Fine-Tuning #############################")
print("######################################################################")
base_model = model.layers[4]
base_model.trainable = True
print(f"Let's see how many layers there are in the base model: {len(base_model.layers)}")

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the 'fine_tune_at' layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = True

learning_rate = base_learning_rate * 0.1
print(f"Chosen learning rate for fine-tuning: {learning_rate}")
print("Chosen optimizer for fine-tuning: Adam\n")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model re-compilation was done!")
fine_tuning_epochs = 5
total_epochs = initial_epochs + fine_tuning_epochs
print(f"Number of epochs for fine-tuning: {fine_tuning_epochs}")
history_fine = model.fit(train_dataset,
                         validation_data=validation_dataset,
                         initial_epoch=history.epoch[-1],
                         epochs=total_epochs)
print("######################### Fine-Tuning - Done #########################\n")

print("######################################################################")
print("######################## Fine-tuning results #########################")
print("######################################################################")
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0,1])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label="Start Fine-Tuning")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0,1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label="Start Fine-Tuning")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Graph printing
tf.keras.utils.plot_model(model, show_shapes=True, to_file="total_model.png", expand_nested=True)

print("##################### Fine-tuning results - Done #####################\n")

saved_model_path = os.path.abspath('../model/MobileNet-V2')
model.save(saved_model_path)
print(f"The MobileNet-V2 trained model was saved successfully in {saved_model_path}")