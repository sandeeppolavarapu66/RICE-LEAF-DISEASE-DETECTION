import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


dataset_path = r'C:\Users\imran\Downloads\drive-download-20230731T141126Z-001'


img_width, img_height = 150, 150


num_classes = 9


batch_size = 32
epochs = 15


datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)


train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)


class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())
class_names = [
    "Hispa",
    "bacterial_leaf_blight",
    "leaf_blast",
    "Brown_spot",
    "Healthy",
    "Shath_Blight",
    "leaf_scald",
    "narrow_brown_spot",
    "Tungro"
]

def build_resnet_model(hp):
    base_model = ResNet50(include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_vgg19_model(hp):
    base_model = VGG19(include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Use Keras Tuner for hyperparameter tuning
tuner_resnet = RandomSearch(
    build_resnet_model,
    objective='val_accuracy',
    max_trials=5,  # Try 5 different hyperparameter configurations
    executions_per_trial=2,  # Train 2 models for each trial to reduce variance
    directory='hyperparameter_tuning_resnet',
    project_name='rice_leaf_disease_resnet'
)
# Search for the best hyperparameter configuration for ResNet
tuner_resnet.search(
    train_generator,
    epochs=epochs,
    validation_data=train_generator,
    validation_steps=train_generator.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)]
)

# Get the best ResNet model and retrain it on the full training data
best_model_resnet = tuner_resnet.get_best_models(num_models=1)[0]

# Fine-tune the last few layers of ResNet
best_model_resnet.layers[0].trainable = True
best_model_resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Train ResNet with early stopping
best_model_resnet.fit(
    train_generator,
    epochs=epochs,
    validation_data=train_generator,
    validation_steps=train_generator.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)]
)

# Use Keras Tuner for hyperparameter tuning for VGG19
tuner_vgg19 = RandomSearch(
    build_vgg19_model,
    objective='val_accuracy',
    max_trials=5,  # Try 5 different hyperparameter configurations
    executions_per_trial=2,  # Train 2 models for each trial to reduce variance
    directory='hyperparameter_tuning_vgg19',
    project_name='rice_leaf_disease_vgg19'
)

# Search for the best hyperparameter configuration for VGG19
tuner_vgg19.search(
    train_generator,
    epochs=epochs,
    validation_data=train_generator,
    validation_steps=train_generator.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)]
)

# Get the best VGG19 model and retrain it on the full training data
best_model_vgg19 = tuner_vgg19.get_best_models(num_models=1)[0]

# Fine-tune the last few layers of VGG19
best_model_vgg19.layers[0].train