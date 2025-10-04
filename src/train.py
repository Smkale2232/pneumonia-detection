# src/train.py - IMPROVED VERSION
import tensorflow as tf
import os


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.has_full_metrics = True  # Track if we have precision/recall/auc metrics

    def get_metrics(self):
        """Define metrics that can be serialized"""
        return [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]

    def compile_model(self, learning_rate=None, full_metrics=True):
        """Compile the model with optional metrics"""
        if learning_rate is None:
            learning_rate = self.config['model']['learning_rate']

        if full_metrics:
            metrics = self.get_metrics()
            self.has_full_metrics = True
        else:
            metrics = ['accuracy']  # Basic metrics only
            self.has_full_metrics = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=metrics
        )

    def setup_callbacks(self):
        """Setup callbacks based on available metrics"""
        # Use different monitoring based on available metrics
        if self.has_full_metrics:
            monitor = 'val_auc'
            mode = 'max'
        else:
            monitor = 'val_accuracy'
            mode = 'max'

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            self.config['paths']['model_save'],
            save_best_only=True,
            monitor=monitor,
            mode=mode,
            verbose=1
        )

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=self.config['training']['patience'],
            restore_best_weights=True,
            mode=mode,
            verbose=1
        )

        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    def train(self, train_ds, val_ds, class_weights=None):
        """Initial training with full metrics"""
        print("=== Phase 1: Training with frozen base model ===")

        self.compile_model(full_metrics=True)
        callbacks = self.setup_callbacks()

        fit_args = {
            'x': train_ds,
            'epochs': self.config['training']['epochs'],
            'validation_data': val_ds,
            'callbacks': callbacks,
            'verbose': 1
        }

        if class_weights is not None:
            fit_args['class_weight'] = class_weights

        history = self.model.fit(**fit_args)
        return history

    def fine_tune(self, base_model, train_ds, val_ds, class_weights, initial_epoch):
        """Fine-tuning with basic metrics to avoid serialization issues"""
        print("=== Phase 2: Fine-tuning ===")

        # Unfreeze layers
        base_model.trainable = True
        for layer in base_model.layers[:-self.config['training']['unfreeze_layers']]:
            layer.trainable = False

        # Recompile with basic metrics only
        self.compile_model(
            learning_rate=self.config['model']['fine_tune_learning_rate'],
            full_metrics=False  # Use basic metrics to avoid issues
        )

        callbacks = self.setup_callbacks()

        history_fine = self.model.fit(
            train_ds,
            epochs=initial_epoch + self.config['training']['fine_tune_epochs'],
            initial_epoch=initial_epoch,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return history_fine
