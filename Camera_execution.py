

# In[1]:


import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import face_recognition
from tensorflow import keras


# In[2]:


os.chdir('C:/APPU Disertation/')


# In[3]:


eye_model = keras.models.load_model('bestModel80.h5')


# In[4]:



# Data Preprocessing Function
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # CONVERT TO GRAYSCALE
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

# webcam frame is inputted into function
def eye_cropper(frame):


    # create a variable for the facial feature coordinates
    facial_features_list = face_recognition.face_landmarks(frame)

    # create a placeholder list for the eye coordinates
    # and append coordinates for eyes to list unless eyes
    # weren't found by facial recognition
    try:
        eye = facial_features_list[0]['left_eye']
    except:
        try:
            eye = facial_features_list[0]['right_eye']
        except:
            return

    # establish the max x and y coordinates of the eye
    x_max = max([coordinate[0] for coordinate in eye])
    x_min = min([coordinate[0] for coordinate in eye])
    y_max = max([coordinate[1] for coordinate in eye])
    y_min = min([coordinate[1] for coordinate in eye])

    # establish the range of x and y coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min

    # in order to make sure the full eye is captured,
    # calculate the coordinates of a square that has a
    # 50% cushion added to the axis with a larger range and
    # then match the smaller range to the cushioned larger range
    if x_range > y_range:
        right = round(.5*x_range) + x_max
        left = x_min - round(.5*x_range)
        bottom = round((((right-left) - y_range))/2) + y_max
        top = y_min - round((((right-left) - y_range))/2)
    else:
        bottom = round(.5*y_range) + y_max
        top = y_min - round(.5*y_range)
        right = round((((bottom-top) - x_range))/2) + x_max
        left = x_min - round((((bottom-top) - x_range))/2)

    # crop the image according to the coordinates determined above
    cropped = frame[top:(bottom + 1), left:(right + 1)]

    # resize the image
    image_for_prediction = cv2.resize(cropped, (80,80))


    return image_for_prediction

# initiate webcam
cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Check if the webcam opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0

# Initialize variables to keep track of true positives, true negatives, false positives, and false negatives
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Create a loop that runs while the webcam is in use
while True:
    # Capture frames being outputted by the webcam
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        break

    try:
        # Perform eye cropping (define eye_cropper function)
        image_for_prediction = eye_cropper(frame)

        # Check if the cropped image is valid
        if image_for_prediction is None:
            continue

        # Image Preprocessing (define preprocessing function)
        image_for_prediction = preprocessing(image_for_prediction)
        image_for_prediction = image_for_prediction.reshape(-1, 80, 80, 1)

        # Get prediction from model (define eye_model)
        prediction = eye_model.predict(image_for_prediction)
        print(prediction)

        # Determine the ground truth (actual state) based on the user's action
        ground_truth = 1  # Adjust this based on the user's behavior (e.g., closed eyes)

        # Based on prediction, update evaluation metrics
        if prediction < 0.2:  
            if ground_truth == 1:  
                true_positives += 1
            else: 
                false_positives += 1
        else:  
            if ground_truth == 1:  
                false_negatives += 1
            else:  
                true_negatives += 1

        # Calculate and display accuracy in real-time
        total_predictions = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total_predictions * 100
        print(f"Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print("Error:", e)


    # Based on prediction, display either "Open Eyes" or "Closed Eyes"
    counter = 0
    
    if  np.max(prediction) < 0.5:
        counter = 0 
        status = 'Eyes Open'

        cv2.rectangle(frame, (round(w/2) - 110,20), (round(w/2) + 110, 80), (38,38,38), -1)

        cv2.putText(frame, status, (round(w/2)-80,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)
        x1, y1,w1,h1 = 0,0,175,75
        ## Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1-20, y1+h1-20), (0,0,0), -1)
        ## Add text
        cv2.putText(frame, 'Active', (x1 +int(w1/10), y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0),2)
    else:
        counter = counter + 1
        status = 'Eyes Closed'

        cv2.rectangle(frame, (round(w/2) - 110,20), (round(w/2) + 110, 80), (38,38,38), -1)

        cv2.putText(frame, status, (round(w/2)-104,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_4)
        x1, y1,w1,h1 = 0,0,175,75
        ## Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1-20, y1+h1-20), (0,0,0), -1)
        ## Add text
        cv2.putText(frame, 'Active', (x1 +int(w1/10), y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0),2)

        # if the counter is greater than 3, play and show alert that user is asleep
        if counter > 2:

            ## Draw black background rectangle
            cv2.rectangle(frame, (round(w/2) - 160, round(h) - 200), (round(w/2) + 160, round(h) - 120), (0,0,255), -1)
            cv2.putText(frame, 'DRIVER SLEEPING', (round(w/2)-136,round(h) - 146), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_4)
            cv2.imshow('Drowsiness Detection', frame)
            k = cv2.waitKey(1)
            
            counter = 0
            continue

    # Resize the frame to the desired width and height
    desired_width = 640  # Adjust this to your preferred width
    desired_height = 480  # Adjust this to your preferred height
    frame = cv2.resize(frame, (desired_width, desired_height))
    
    cv2.imshow('Drowsiness Detection', frame)
    k = cv2.waitKey(1)
    if k == ord(' '): # PRESS SPACEBAR TO EXIT THE WINDOW(PREDICTION)
        break

cap.release()
cv2.destroyAllWindows()

