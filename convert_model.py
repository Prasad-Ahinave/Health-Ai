from tensorflow.keras.models import load_model

# Load your existing model
model = load_model("skin_model.h5")

# Save it in the new Keras format
model.save("skin_model.keras")

print("Model converted successfully!")