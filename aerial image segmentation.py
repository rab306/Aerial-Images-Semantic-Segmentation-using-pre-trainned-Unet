# Importing required libraries
import os
import cv2
import numpy as np


import matplotlib.pyplot as plt
from patchify import patchify
from PIL import Image

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import jaccard_loss, DiceLoss
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K

# Images preprocessing
scaler = MinMaxScaler()  # We will scale the images not the masks
directory = 'Semantic segmentation dataset/'

patch_size = 256
image_dataset = []
for path, subdirs, files in os.walk(directory):
    dirname = path.split(os.path.sep)[-1]
    # print(dirname)
    if dirname == 'images':
        images = os.listdir(path)
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"):

                image = cv2.imread(path + "/" + image_name, 1)  # Equal to cv2.IMREAD_COLOR
                x_size = (image.shape[1] // patch_size) * patch_size
                y_size = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, x_size, y_size))
                image = np.array(image)
                print(image.shape)
                # Extract patches from each image with no overlap
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                print(patches_img.shape)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]
                        print(f'before scaling : {single_patch_img.shape}')
                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        print(f'after scaling : {single_patch_img.shape}')   # Checking the consistency of the image data
                        # Dropping the extra unecessary dimension that patchify adds.
                        single_patch_img = single_patch_img[0]
                        image_dataset.append(single_patch_img)

# mMsk preprocessing
patch_size = 256
mask_dataset = []
for path, subdirs, files in os.walk(directory):
    dirname = path.split(os.path.sep)[-1]
    print(dirname)
    if dirname == 'masks':
        masks = os.listdir(path)
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".png"):
                mask = cv2.imread(path + "/" + mask_name, 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                x_size = (mask.shape[1] // patch_size) * patch_size
                y_size = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, x_size, y_size))
                mask = np.array(mask)

                # extract patches with no overlap
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                # print(patches_mask.shape)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]

                        # Dropping the extra unecessary dimension that patchify adds.
                        single_patch_mask = single_patch_mask[0]
                        mask_dataset.append(single_patch_mask)



# Converting the image and masks lists into arrays
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)
print(image_dataset.shape, mask_dataset.shape)

# Defining hexadecimal colors for each class
class_colors_hex = {
    'Building': '#3C1098',
    'Land': '#8429F6',
    'Road': '#6EC1E4',
    'Vegetation': '#FEDD3A',
    'Water': '#E2A929',
    'Unlabeled': '#9B9B9B'
}

# Converting hexadecimal colors to integer labels
def hex_to_int(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))

class_colors = {key: hex_to_int(value) for key, value in class_colors_hex.items()}

# Converting RGB colors to integer labels
def rgb_to_2D_label(label):
    """
    Supply our label masks as input in RGB format.
    Replace pixels with specific RGB values...
    """
    label_seg = np.zeros(label.shape[:-1], dtype=np.uint8)
    for idx, color in enumerate(class_colors.values()):
        label_seg[np.all(label == color, axis=-1)] = idx
    return label_seg

# Converting all masks in mask_dataset to integer labels
labels = np.array([rgb_to_2D_label(mask) for mask in mask_dataset])
labels = np.expand_dims(labels, axis=3)

print( np.unique(labels), labels.shape, image_dataset.shape)


# Checking the consistency between image data and labeled data
fig = plt.figure(figsize=(12,10))
ax = fig.subplot_mosaic("""CD""")
ax['C'].imshow(image_dataset[10])
ax['D'].imshow(labels[10])
plt.show()


# Converting labels to categories
num_classes = len(class_colors)
labels_categ = to_categorical(labels, num_classes=num_classes)
print(labels_categ.shape)

# Splitting the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_categ, test_size = 0.20, random_state = 42)

# Data Augmentation
# Creating the data generators
batch_size = 16

data_gen_args = dict(
    rotation_range=45.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
image_data_gen = ImageDataGenerator(**data_gen_args)
label_data_gen = ImageDataGenerator(**data_gen_args)


def data_generator(image_data_gen, label_data_gen, X_train, y_train, batch_size):
    image_generator = image_data_gen.flow(X_train, batch_size=batch_size, seed=42)
    label_generator = label_data_gen.flow(y_train, batch_size=batch_size, seed=42)
    while True:
        X_batch = next(image_generator)
        y_batch = next(label_generator)

        yield X_batch, y_batch

train_generator = data_generator(image_data_gen, label_data_gen, X_train, y_train, batch_size)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define custom loss functions
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def jaccard_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def combined_loss(y_true, y_pred):
    alpha = 0.34
    beta = 0.33
    gamma = 0.33

    cat_loss = categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    jaccard = jaccard_loss(y_true, y_pred)

    return alpha * cat_loss + beta * dice + gamma * jaccard


# Loading pre-trained U-Net model (for example purposes)
base_model = sm.Unet(backbone_name='resnet34', encoder_weights='imagenet')

# Freezing model
for layer in base_model.layers:
    layer.trainable = False
# Try unfreezing some of the high level layers and fine tune their parameters

# Modifing the output layer for the number of classes
num_classes = len(np.unique(labels))
x = base_model.output
x = tf.keras.layers.Conv2D(num_classes, (1, 1))(x)
x = tf.keras.layers.Activation('softmax')(x)

# Creating the model
model = Model(inputs=base_model.input, outputs=x)

# Compiling step
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)

# Plotting training & validation loss/accuracy
plt.figure(figsize=(12, 5))

# Plotting loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
