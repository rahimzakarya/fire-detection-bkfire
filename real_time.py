# open camera using opencv
import cv2
import numpy as np

# load model
from tensorflow.keras.models import load_model
model = load_model('trainedModel/Bkfire-Model-1.h5')

# define video capture object
cap = cv2.VideoCapture(0)

# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 255, 0)

# define labels
labels_dict = {0:'FIRE', 1:'NO FIRE'}

# Data preprocess
size = 64
def preprocess(X):
    X = np.array(X).reshape(-1, size, size, 3)
    X = X/255.0
    return X


# Start capturing the feed
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_final = frame.copy()
    # resize the captured frame
    frame = cv2.resize(frame, (size, size))
    # reshape the image to support our model input and normalizing
    image = preprocess(frame)
    # predict
    pred = model.predict(image)
    # get the index of the maximum prediction
    pred = np.argmax(pred)
    # get the label
    label = labels_dict[pred]

    # display the label
    cv2.putText(frame_final, label, (50, 50), font, 3, text_color, 2)

    # display the output
    cv2.imshow('Fire Detection', frame_final)
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()