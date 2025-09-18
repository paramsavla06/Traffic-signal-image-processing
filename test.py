import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# signnames.csv ka path (dataset ke folder me hona chahiye)
csv_path = "C:/Users/param/Downloads/python/tih project/label_names.csv"

signnames = pd.read_csv(csv_path)
class_dict = dict(zip(signnames.ClassId, signnames.SignName))

print("Total classes:", len(class_dict))
for i in class_dict:
    print(f"{i}: {class_dict[i]}")

# Load model
model = load_model("final_trained_model.keras")

# Load saved mapping
with open("class_mapping.pkl", "rb") as f:
    index_to_class = pickle.load(f)

print("Mapping loaded:", index_to_class)

# Load test image
img_path = "traffic_sign.jpg"
img = image.load_img(img_path, target_size=(32, 32))

# Convert image to array
img_array = image.img_to_array(img)
img_array = img_array.astype("float32") / 255.0  # normalize
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = index_to_class[predicted_index]
print(prediction)

# Actual class (manually known)
actual_class = 5

# Show result
plt.imshow(img)
plt.axis("off")
plt.title(f"Actual: {actual_class} | Predicted: {predicted_class}")
plt.show()

print(f"Actual class: {actual_class}")
print(f"Predicted class: {predicted_class}")
