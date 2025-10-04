# src/grad_cam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Set last conv layer name based on base model
        base_model_name = config['model']['base_model']
        if base_model_name == 'MobileNetV2':
            self.last_conv_layer_name = "Conv_1"
        elif base_model_name == 'EfficientNetB0':
            self.last_conv_layer_name = "top_conv"
        else:  # VGG16
            self.last_conv_layer_name = "block5_conv3"

    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """Step 6: Generate Grad-CAM heatmap"""
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(
                self.last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def display_gradcam(self, img, heatmap, alpha=0.4):
        """Step 6: Display Grad-CAM overlay"""
        heatmap = np.uint8(255 * heatmap)

        jet = plt.colormaps.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

        return superimposed_img

    def visualize_for_batch(self, test_dataset, num_images=6):
        """Step 6: Visualize Grad-CAM for batch"""
        for images, labels in test_dataset.take(1):
            break

        indices = np.random.choice(len(images), min(
            num_images, len(images)), replace=False)

        fig, axes = plt.subplots(2, num_images, figsize=(20, 8))

        for i, idx in enumerate(indices):
            img = images[idx].numpy()
            true_label = labels[idx].numpy()

            img_array = tf.expand_dims(img, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array)

            pred = self.model.predict(img_array, verbose=0)[0][0]
            pred_class = "Pneumonia" if pred > 0.5 else "Normal"

            heatmap = self.make_gradcam_heatmap(img_array)

            # Original image
            axes[0, i].imshow(img / 255.0)
            axes[0, i].set_title(f'True: {"Pneumonia" if true_label == 1 else "Normal"}\n'
                                 f'Pred: {pred_class} ({pred:.3f})')
            axes[0, i].axis('off')

            # Grad-CAM
            superimposed_img = self.display_gradcam(img, heatmap)
            axes[1, i].imshow(superimposed_img)
            axes[1, i].set_title('Grad-CAM')
            axes[1, i].axis('off')

        plt.tight_layout()
        os.makedirs('results/grad_cam_visualizations', exist_ok=True)
        plt.savefig('results/grad_cam_visualizations/grad_cam_batch.png')
        plt.show()
