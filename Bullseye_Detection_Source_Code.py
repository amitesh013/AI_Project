!pip install kagglehub tensorflow opencv-python matplotlib
!pip install pandas seaborn


import os
import shutil
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


kaggle_path = "/Users/digitalmarketing/.cache/kagglehub/datasets/ashwinsangareddypeta/bulls-eye-target-images-scraped-from-google/versions/2"
project_path = "BULLS EYE/images"

for split in ['train', 'test']:
    for label in ['bulls eye', 'not bulls eye']:
        folder = os.path.join(project_path, split, label)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

image_files = [f for f in os.listdir(kaggle_path + '/train') if f.endswith('.png')]
for idx, img_file in enumerate(image_files):
    src_path = os.path.join(kaggle_path, 'train', img_file)
    shutil.copy2(src_path, os.path.join(project_path, 'train', 'bulls eye', img_file))
    if idx % 4 == 0:
        shutil.copy2(src_path, os.path.join(project_path, 'test', 'bulls eye', img_file))


not_bulls_eye_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
for split in ['train', 'test']:
    folder = os.path.join(project_path, split, "not bulls eye")
    for i in range(4):
        img = Image.fromarray(not_bulls_eye_image)
        img.save(os.path.join(folder, f"dummy_not_bulls_eye_{i+1}.png"))


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(project_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(project_path, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(train_generator, epochs=20, validation_data=test_generator)


eval_generator = test_datagen.flow_from_directory(
    os.path.join(project_path, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

if len(eval_generator)>0:
   try:
       loss, accuracy = model.evaluate(eval_generator)
       print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
   except VaalueError as e:
       print (f"Evalution Error: (e)")


tflite_model_path = "android.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print("✅ Model saved as android.tflite")

def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array.astype(np.float32), axis=0), img

def detect_inner_circle(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=40,
        param2=30,
        minRadius=10,
        maxRadius=25
    )

    h, w = img.shape[:2]
    image_center = np.array([w // 2, h // 2])
    bullseye_accuracy = 0.0

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        closest = min(circles, key=lambda c: np.linalg.norm(image_center - np.array([c[0], c[1]])))

        cv2.circle(img, (closest[0], closest[1]), closest[2], (0, 255, 0), 2)

        dist_to_center = np.linalg.norm(image_center - np.array([closest[0], closest[1]]))
        max_dist = np.linalg.norm(image_center - np.array([0, 0]))
        bullseye_accuracy = (1 - dist_to_center / max_dist) * 100
        print(f"🎯 Bullseye Accuracy: {bullseye_accuracy:.2f}%")
    else:
        print("⭕ No circles detected")

    return img, bullseye_accuracy



interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

bulls_eye_dir = os.path.join(project_path, "test", "bulls eye")
sample_img_path = os.path.join(bulls_eye_dir, os.listdir(bulls_eye_dir)[0])

input_data, original_img = preprocess_image(sample_img_path, (224, 224))
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
pred_class = np.argmax(output)
label = 'bulls eye' if pred_class == 0 else 'not bulls eye'

# Show result with bullseye accuracy
circle_img, accuracy = detect_inner_circle(sample_img_path)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {label}\nBullseye Accuracy: {accuracy:.2f}%")
plt.axis('off')
plt.show()


from sklearn.metrics import classification_report, confusion_matrix


y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)


class_labels = list(test_generator.class_indices.keys())


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_true, y_pred, target_names=class_labels))

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class_labels = ['bulls eye'] * 30 + ['not bulls eye'] * 4
df = pd.DataFrame({'class': class_labels})
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.show()


results_df['Confidence Bucket'] = pd.cut(
    results_df['Confidence'], 
    bins=[0, 0.5, 0.7, 0.85, 1.0], 
    labels=['Low', 'Medium', 'High', 'Very High']
)

bucket_crosstab = pd.crosstab(results_df['True Label'], results_df['Confidence Bucket'])

sns.heatmap(bucket_crosstab, annot=True, cmap='YlGnBu')
plt.title('Prediction Distribution by Confidence Buckets and True Class')
plt.ylabel('True Class')
plt.xlabel('Confidence Bucket')
plt.show()






