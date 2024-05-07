import os
import warnings
from hyperopt import hp, fmin, tpe, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.applications.xception import Xception
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from xgboost import XGBClassifier
import cv2
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# Load images and labels
general_path = "F:/Downloads/archive (10)"
path_train_images = os.path.join(general_path, "train")  # Path to training images
path_test_images = os.path.join(general_path, "test")  # Path to test images

def limit_data(data_dir):
    a = []
    for i in os.listdir(data_dir):
        for j in os.listdir(os.path.join(data_dir, i)):
            a.append((os.path.join(data_dir, i, j), i))
    return pd.DataFrame(a, columns=['filename', 'class'])

# Use the function to limit the data
limited_train_data = limit_data(path_train_images)
limited_test_data = limit_data(path_test_images)

# Data augmentation
datagen = ImageDataGenerator(rescale=1. / 255)
train_dataset = datagen.flow_from_dataframe(dataframe=limited_train_data)
test_dataset = datagen.flow_from_dataframe(dataframe=limited_test_data)

# Now we will try to use Xception Feature extraction architecture
non_trainable_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(train_dataset.image_shape[0], train_dataset.image_shape[1], train_dataset.image_shape[2]))

for layer in non_trainable_model.layers:
    layer.trainable = False

x = non_trainable_model.output
x = tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
flatten = tf.keras.layers.Flatten()(x)
predictions = tf.keras.layers.Dense(7, activation='softmax')(flatten)

model = tf.keras.Model(inputs=non_trainable_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=20 )

model2 = tf.keras.Model(inputs=non_trainable_model.input, outputs=flatten)

model.save('model2.keras')

train_data_flatten = model2.predict(train_dataset)
test_data_flatten = model2.predict(test_dataset)

x_train = train_data_flatten
y_train = np.array(train_dataset.classes)
x_test = test_data_flatten
y_test = np.array(test_dataset.classes)

# Define the search space
search_space = {
    'C': hp.uniform('C', 0.1, 10),  # Uniform distribution for C
    'gamma': hp.choice('gamma', ['scale', 'auto']),  # Categorical choice for gamma
    'kernel': hp.choice('kernel', ['linear', 'rbf']),  # Categorical choice for kernel
    'degree': hp.quniform('degree', 1, 6, 1),  # Integer quantized uniform distribution for degree
    'coef0': hp.uniform('coef0', 0, 1),  # Uniform distribution for coef0
    'shrinking': hp.choice('shrinking', [True, False]),  # Categorical choice for shrinking
    'probability': hp.choice('probability', [True, False]),  # Categorical choice for probability
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),  # Log uniform distribution for tol
    }
kernel_mapping = {'linear': 0, 'rbf': 1}
gamma_mapping = {'scale': 0,'auto': 1}

def objective(params, x_train, y_train, x_test, y_test):
    kernel = ['linear', 'rbf'][kernel_mapping[params['kernel']]]
    gamma = ['scale', 'auto'][gamma_mapping[params['gamma']]]
    shrinking = [True, False][params['shrinking']]
    probability = [True, False][params['probability']]
    
    converted_params = {
        'C': params['C'],
        'kernel':kernel,
        'degree': int(params['degree']),
        'coef0': params['coef0'],
        'shrinking': shrinking,
        'probability': probability,
        'tol': params['tol'],
        'gamma': gamma
    }
    
    svm = SVC(**converted_params)
    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    return -score  # Minimize the negative score for better performance


   
# Optimize hyperparameters using fmin
best_params = fmin(fn=lambda params: objective(params, x_train, y_train, x_test, y_test),
                   space=search_space,
                   algo=tpe.suggest,
                   max_evals=100)

# Train the SVM model with the optimized hyperparameters
converted_best_params = {
    'C': best_params['C'],
    'kernel': ['linear', 'rbf'][best_params['kernel']],
    'degree': int(best_params['degree']),
    'coef0': best_params['coef0'],
    'shrinking': [True, False][best_params['shrinking']],
    'probability': [True, False][best_params['probability']],
    'tol': best_params['tol'],
    'gamma': ['scale', 'auto'][best_params['gamma']]
}

svm = SVC(**converted_best_params)
svm.fit(x_train, y_train)

# Evaluate the SVM model
accuracy_svm = svm.score(x_test, y_test)
print("Accuracy:", accuracy_svm)

# Load the compiled model
model = load_model('model2.keras')

# Start video capture
cap = cv2.VideoCapture(0)

# Define a dictionary to map the class index to the corresponding emotion
emotion_dict = {0: 'Angry', 1: 'Surprised', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Neutral', 6: 'Sad'}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess the frame to match your model's input requirements
    resized_frame = cv2.resize(frame, (256, 256))  # Assuming model input shape is (224, 224, 3)
    preprocessed_frame = resized_frame.astype('float32') / 255
    input_frame = np.expand_dims(preprocessed_frame, axis=0)

    # Use the model to predict the emotion of the person in the frame
    prediction = model.predict(input_frame)

    # Get the emotion from the prediction
    emotion = np.argmax(prediction)
    emotion_str = emotion_dict[emotion]

    # Display the detected emotion on the frame
    cv2.putText(frame, f'Emotion: {emotion_str}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame with the predicted emotion
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows 
cv2.destroyAllWindows()



