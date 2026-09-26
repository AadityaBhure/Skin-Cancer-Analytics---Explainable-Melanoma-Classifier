"""Grad-CAM of melanoma probability for the trained functional model."""
import numpy as np
import tensorflow as tf
from matplotlib import colormaps


def make_heatmap(model, image_batch, class_names):
    candidates = [layer for layer in model.layers
                  if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
                  and len(layer.output.shape) == 4]
    if not candidates:
        raise ValueError("No four-dimensional convolutional layer found in the model.")
    layer = candidates[-1]
    gradient_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    inputs = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # Watch inputs explicitly because the backbone's weights are frozen.
        tape.watch(inputs)
        activations, output = gradient_model(inputs, training=False)
        score = output[:, 0] if class_names.index("melanoma") == 1 else 1.0 - output[:, 0]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("Grad-CAM graph is disconnected from the prediction.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(heatmap[..., None], (224, 224)).numpy()[..., 0]
    if not np.isfinite(heatmap).all():
        raise ValueError("Grad-CAM produced non-finite values.")
    if not np.any(heatmap):
        print("Grad-CAM has no positive activation for melanoma in this image; saving a zero heatmap.")
    return heatmap, layer.name


def overlay_heatmap(original, heatmap, alpha=0.40):
    colored = colormaps["jet"](heatmap)[..., :3] * 255.0
    return np.clip((1 - alpha) * np.asarray(original, dtype=np.float32) + alpha * colored, 0, 255).astype(np.uint8)
