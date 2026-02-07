import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(
    "models/cnn_baseline.h5",
    compile=False
)

# Export as TensorFlow SavedModel (Keras 3 compatible)
model.export("models/cnn_baseline_tf")

print(" Model exported in TensorFlow SavedModel format")
