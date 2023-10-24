import cv2
import numpy as np
import os
from PIL import Image

# Path for face image database
dataset_path = 'dataset'

# For OpenCV 4.x and later
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(dataset_path):
    faceSamples = []
    labels = []
    label_dict = {}  # Dictionary to map user names to unique integer labels

    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            user_folder = os.path.join(root, dir_name)

            # Assign a unique integer label to each user (directory)
            if dir_name not in label_dict:
                label_dict[dir_name] = len(label_dict)

            label = label_dict[dir_name]
            image_paths = [os.path.join(user_folder, f) for f in os.listdir(user_folder)]

            for image_path in image_paths:
                PIL_img = Image.open(image_path).convert('L')  # Convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                faceSamples.append(img_numpy)
                labels.append(label)

    return faceSamples, labels

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, labels = getImagesAndLabels(dataset_path)

# Convert the labels list to a NumPy array of type int32
labels = np.array(labels, dtype=np.int32)

recognizer.train(faces, labels)

# Save the model into trainer/trainer.yml
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer.write(os.path.join(trainer_path, 'trainer.yml'))

# Print the number of faces trained and end the program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(labels))))
