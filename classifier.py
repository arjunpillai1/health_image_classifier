#imports
import os
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


# -----------------------------------------
# Preprocess
# -----------------------------------------
base_dir = "Images_individual"

# Structure
# Images_individual/
#     class1/
#     class2/

# MAKE SURE THERE'S NO OVERLAP BETWEEN TRAINING AND VALIDATION WHEN SPLIT
seed = 42 
# original images (rescale and resize only)
original_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  
    validation_split=0.2,  # Keep validation split consistent
)

# augmented images 
augmented_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

batch_size = 4

# Original generator (no augmentations)
original_train_generator = original_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed,
)

# generator with augmentations
augmented_train_generator = augmented_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed,
)

# Validation uses original images
validation_generator = original_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed,
)


validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

# -----------------------------------------
# View validation images before training
# -----------------------------------------
def plot_images(images, labels, class_labels, title=""):
    plt.figure(figsize=(15, 15))
    for i in range(len(images)):
        plt.subplot(5, 5, i + 1)  
        plt.imshow(images[i])
        true_label = class_labels[labels[i]]
        plt.title(f"True: {true_label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


class_labels = {v: k for k, v in original_train_generator.class_indices.items()}

validation_images = []
validation_labels = []

for i in range(validation_steps):
    batch_images, batch_labels = next(validation_generator)
    validation_images.extend(batch_images)
    validation_labels.extend(np.argmax(batch_labels, axis=1))

plot_images(validation_images, validation_labels, class_labels, title="Validation Set Images")



# -----------------------------------------
# Class weights and model setup
# -----------------------------------------

class_indices = original_train_generator.class_indices
classes = list(class_indices.keys())
num_classes = len(classes)

class_counts = original_train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_counts),
    y=class_counts,
)
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# Load  model without  top layers
conv_base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)

conv_base.trainable = False

#additional layers
model = Sequential()
model.add(conv_base)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  

# Compile 
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

def combined_generator(original_generator, augmented_generator, augment_ratio=1):
    """
    Combines batches from original and augmented generators with a specified ratio of augmented to original images.
    augment_ratio: The number of augmented batches to include for every batch of original images.
    """
    while True:
        
        original_batch_images, original_batch_labels = next(original_generator)
        
        # Initialize the combined batch 
        combined_images = original_batch_images
        combined_labels = original_batch_labels
        
        # Add augmented images per augment_ratio
        for _ in range(augment_ratio):
            augmented_batch_images, augmented_batch_labels = next(augmented_generator)
            combined_images = np.concatenate((combined_images, augmented_batch_images), axis=0)
            combined_labels = np.concatenate((combined_labels, augmented_batch_labels), axis=0)
        
        yield combined_images, combined_labels

# set for original:aug
augment_ratio = 2  

train_generator = combined_generator(original_train_generator, augmented_train_generator, augment_ratio)

# -----------------------------------------
# View random images from the training set 
# -----------------------------------------
def sample_images_from_generator(generator, num_images):
    images = []
    labels = []
    count = 0
    while count < num_images:
        batch_images, batch_labels = next(generator)
        for i in range(len(batch_images)):
            images.append(batch_images[i])
            labels.append(np.argmax(batch_labels[i]))
            count += 1
            if count >= num_images:
                break
    return images, labels


# Plot 10 
# sampled_images, sampled_labels = sample_images_from_generator(train_generator, 10)
# plot_images(sampled_images, sampled_labels, class_labels, title="Training Set Sample Images")

# Update steps_per_epoch based on the augment_ratio
steps_per_epoch = max(1, (original_train_generator.samples * (1 + augment_ratio)) // batch_size)  # Adjusted for batch size

# Val steps stay same

# -----------------------------------------
# Training
# -----------------------------------------

# Reset Generators

validation_generator.reset()

num_epochs = 20

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,  # handle imbalance
)

# Save model
# model_name = f"image_classifier_MobileNetV2_epoch{num_epochs}_batchsize{batch_size}_trainingsamples{original_train_generator.samples}.keras"
model_name = "health_image_classifier.keras"
model.save(model_name)


# -----------------------------------------
# Analysis
# -----------------------------------------
# Predictions on the validation set
validation_generator.reset()
val_steps = validation_generator.samples // batch_size + 1
Y_pred = model.predict(validation_generator, steps=val_steps)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = validation_generator.classes

report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    zero_division=0,
)
print("Classification Report:")
print(report)

# Plot accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(f"MobileNetV2_epoch{num_epochs}_batchsize{batch_size}_trainingsamples{original_train_generator.samples}.png")
plt.show()


# Visualizing Some Predictions

validation_generator.reset()
val_images, val_labels = next(validation_generator)

# Predict on the batch
predictions = model.predict(val_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(val_labels, axis=1)

class_labels = {v: k for k, v in class_indices.items()}

# Display with true and predicted labels
plt.figure(figsize=(15, 15))
for i in range(min(16, len(val_images))):
    plt.subplot(4, 4, i + 1)
    plt.imshow(val_images[i])
    true_label = class_labels[true_classes[i]]
    predicted_label = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()