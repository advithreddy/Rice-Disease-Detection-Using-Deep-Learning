import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import cv2
import matplotlib.pyplot as plt

# Set your dataset path
dataset_path = "C:\\Users\\advit\\OneDrive\\Desktop\\Data_Set"

# Set your output path for saving the model
output_path = 'C:\\Users\\advit\\OneDrive\\Desktop\\Rice_Disease_Detection'

# Function to create the model
def create_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Function to compile and train the model
def train_model(model, train_path, validation_path, output_path, epochs=10, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                                  target_size=(224, 224),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        verbose=1)

    model.save(os.path.join(output_path, 'rice_disease_model.h5'))

    return history

# Function to generate random photos for training and validation
def generate_random_photos(base_path, num_photos=10):
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        
        for i in range(num_photos):
            # Generate a random image
            image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

            # Save the image to the class directory
            img_path = os.path.join(class_path, f"random_image_{i}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Number of classes (disease types)
    num_classes = 3

    # Create model
    input_shape = (224, 224, 3)
    model = create_model(input_shape, num_classes)

    # Ensure 'train' and 'validation' directories exist
    train_path = "C:\\Users\\advit\\OneDrive\\Desktop\\Rice_Disease_Detection\\train"
    validation_path = "C:\\Users\\advit\\OneDrive\\Desktop\\Rice_Disease_Detection\\validation"
    
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    # Generate random photos for training and validation
    generate_random_photos(train_path)
    generate_random_photos(validation_path)

    # Train the model
    history = train_model(model, train_path, validation_path, output_path, epochs=100, batch_size=32)

    # Visualize training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
