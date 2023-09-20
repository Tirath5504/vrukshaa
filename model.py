import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define the paths for your preprocessed dataset
preprocessed_root = '/home/tirath/Documents/vruksha/vrukshaa/preprocessed_dataset'

# Define the model and training parameters
input_shape = (224, 224, 3)
num_classes = len(os.listdir(os.path.join(preprocessed_root, 'train')))  # Number of disease classes

# Data augmentation (optional)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the pre-trained VGG-19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom classification layers on top of VGG-19
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the layers of the pre-trained VGG-19 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    os.path.join(preprocessed_root, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    os.path.join(preprocessed_root, 'validation'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('/home/tirath/Documents/vruksha/vrukshaa/model.h5')

print("Model training complete.")
