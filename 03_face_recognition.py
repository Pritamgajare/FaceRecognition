import cv2
import os

# Path for face image database
path = 'dataset'

# For OpenCV 4.x and later
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Lower the confidence threshold for recognition
confidence_threshold = 80
min_probability = 40

# Create a mapping dictionary to associate integer labels with user names
label_to_name = {}

# Function to get user name by label
def get_name_by_label(label):
    return label_to_name[label] if label in label_to_name else "unknown"

# Read the directories in 'dataset' to populate the label_to_name dictionary
for root, dirs, files in os.walk(path):
    for dir_name in dirs:
        user_folder = os.path.join(root, dir_name)
        image_paths = [os.path.join(user_folder, f) for f in os.listdir(user_folder)]
        
        if not image_paths:
            continue

        # Read the first image in the folder to get the user name
        image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        label, _ = recognizer.predict(image)

        label_to_name[label] = dir_name

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is below the threshold and above the minimum probability
        if confidence < confidence_threshold and (100 - confidence) > min_probability:
            name = get_name_by_label(id)
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
