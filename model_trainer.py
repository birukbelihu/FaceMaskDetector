import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set Local Paths
data_path = "dataset"
model_save_path = "face_mask_detector.h5"
tflite_save_path = "face_mask_detector.tflite"
test_image_file_path = "test_1.jpg"

def prepare_data():
    image_size = (224, 224)
    batch_size = 32
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        data_path, target_size=image_size, batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_path, target_size=image_size, batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )
    return train_gen, val_gen

def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_gen, val_gen, epochs=10):
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return history

def save_model(model):
    model.save(model_save_path)

def predict_on_image(model, class_indices):
    test_img = load_img(test_image_file_path, target_size=(224, 224))
    img_array = img_to_array(test_img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]
    class_labels = {v: k for k, v in class_indices.items()}
    print(f"Prediction: {class_labels[predicted_index]} ({confidence * 100:.1f}%)")

def convert_to_tensorflow_lite():
    model = load_model(model_save_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_save_path, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    print(os.listdir(data_path))
    train_gen, val_gen = prepare_data()
    model = build_model()
    model.summary()
    train_model(model, train_gen, val_gen, epochs=10)
    save_model(model)
    predict_on_image(model, train_gen.class_indices)
    # Uncomment to convert to The Model TensorFlow Lite
    # convert_to_tensorflow_lite()