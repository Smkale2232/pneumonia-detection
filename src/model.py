# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models


def create_augmentation_layer():
    """Step 2: Data Augmentation"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def create_pneumonia_model(config):
    """Step 3: Transfer Learning Model"""
    base_model_name = config['model']['base_model']

    # Input layer
    inputs = tf.keras.Input(shape=(*config['model']['img_size'], 3))

    # Data augmentation
    x = create_augmentation_layer()(inputs)

    # Preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model selection
    if base_model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*config['model']['img_size'], 3),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*config['model']['img_size'], 3),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(
            input_shape=(*config['model']['img_size'], 3),
            include_top=False,
            weights='imagenet'
        )

    # Freeze base model
    base_model.trainable = False

    # Pass through base model
    x = base_model(x, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model
