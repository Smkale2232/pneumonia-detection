# src/data_loader.py - FIXED VERSION
import tensorflow as tf
import numpy as np


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.img_size = config['model']['img_size']
        self.batch_size = config['model']['batch_size']

    def load_datasets(self):
        """Load datasets with memory optimizations"""
        print("Loading training dataset...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config['data']['train_dir'],
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='binary'
        )

        print("Loading validation dataset...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config['data']['val_dir'],
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='binary'
        )

        print("Loading test dataset...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.config['data']['test_dir'],
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='binary',
            shuffle=False
        )

        return train_ds, val_ds, test_ds

    def calculate_class_weights(self, dataset):
        """Calculate class weights for imbalance - FIXED VERSION"""
        print("Calculating class weights...")
        class_counts = [0, 0]  # [normal, pneumonia]

        for images, labels in dataset:
            # Convert TensorFlow tensors to numpy for counting
            labels_np = labels.numpy()
            class_counts[0] += np.sum(labels_np == 0)
            class_counts[1] += np.sum(labels_np == 1)

        total = class_counts[0] + class_counts[1]
        print(
            f"Class counts - Normal: {class_counts[0]}, Pneumonia: {class_counts[1]}")

        # Calculate weights using numpy floats
        weight_for_0 = total / (2.0 * class_counts[0])
        weight_for_1 = total / (2.0 * class_counts[1])

        class_weights = {0: weight_for_0, 1: weight_for_1}
        print(f"Class weights: {class_weights}")

        return class_weights

    def optimize_datasets(self, train_ds, val_ds, test_ds):
        """Optimize datasets for GTX 1650 memory"""
        AUTOTUNE = tf.data.AUTOTUNE

        # Use caching and prefetching for better performance
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds
