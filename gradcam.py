"""
gradcam.py — Grad-CAM heatmap generation for the DeepFER MobileNetV2 model.

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions
of the face the model focused on when making its prediction — useful for
sanity-checking that the model is looking at eyes/mouth/brows rather than
background or irrelevant features.

Usage:
    from gradcam import GradCAM

    cam = GradCAM(model)  # auto-detects the last conv layer
    heatmap = cam.compute(preprocessed_image_batch, class_index)
    overlay = cam.overlay_on_image(original_bgr_image, heatmap)
"""

import numpy as np
import tensorflow as tf
import cv2


class GradCAM:
    """Computes Grad-CAM heatmaps for a Keras classification model."""

    def __init__(self, model, layer_name=None):
        """
        Args:
            model: the loaded Keras model (e.g. from EmotionClassifier.model).
            layer_name: name of the conv layer to use. If None, auto-detects
                the last layer with 4D output (spatial feature map) before
                the pooling/dense head — this is "out_relu" for the DeepFER
                MobileNetV2 architecture.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()

        # Build a model that outputs both the target conv layer's activations
        # and the final predictions, so we can compute gradients between them.
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(self.layer_name).output, model.output],
        )

    def _find_last_conv_layer(self):
        """Walk backwards through the model to find the last 4D-output layer."""
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError(
            "Could not auto-detect a conv layer with 4D output. "
            "Pass layer_name explicitly, e.g. GradCAM(model, layer_name='out_relu')."
        )

    def compute(self, preprocessed_batch, class_index=None):
        """Compute the Grad-CAM heatmap for one image.

        Args:
            preprocessed_batch: a (1, 224, 224, 3) array — already run through
                mobilenet_v2's preprocess_input, same as what you'd feed to
                model.predict().
            class_index: which class to explain. If None, uses the model's
                own top prediction (most common use case).

        Returns:
            heatmap: a (H, W) float32 array, values normalized to [0, 1],
                where H/W match the conv layer's spatial size (e.g. 7x7 for
                MobileNetV2 at 224x224 input — gets resized when overlaid).
        """
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(preprocessed_batch)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, class_index]

        # Gradient of the target class score w.r.t. the conv layer's output —
        # this tells us how much each spatial location/channel mattered.
        grads = tape.gradient(class_channel, conv_output)

        # Average the gradients over width/height to get one importance
        # weight per channel, then weight the conv output channels by that
        # and sum — this is the core Grad-CAM operation.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU — we only care about features that positively influenced the
        # predicted class, not ones that pushed away from it.
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy(), class_index

    def overlay_on_image(self, original_bgr_image, heatmap, alpha=0.45):
        """Overlay a Grad-CAM heatmap on the original face image.

        Args:
            original_bgr_image: the original face crop (BGR, any size — this
                is what gets shown, so pass the un-preprocessed crop the user
                will recognize, not the normalized model input).
            heatmap: output from compute().
            alpha: blend strength of the heatmap overlay (0 = invisible,
                1 = fully opaque heatmap, original image hidden).

        Returns:
            BGR image (same size as original_bgr_image) with a jet-colormap
            heatmap overlay showing where the model focused.
        """
        h, w = original_bgr_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_bgr_image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay
