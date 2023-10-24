import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the name of the user
name = input("\nEnter the name of the user: ")

# Create a folder for the user if it doesn't exist
user_folder = os.path.join("dataset", name)
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 1

while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the captured image into the user's folder with the user's name and count
        image_path = os.path.join(user_folder, f"{name}_{count}.jpg")
        cv2.imwrite(image_path, gray[y:y + h, x:x + w])
        count += 1

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count > 30:  # Take 30 face samples and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
