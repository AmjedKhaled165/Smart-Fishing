from keras.models import load_model
import cv2
import numpy as np
import time

np.set_printoptions(suppress=True)
model = load_model("Smart Fishing Model.h5", compile=False)
class_names = ["Fish", "Shark", "Diver", "Trash"]
camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Type:", class_name[0:], end="")
    print(" , Accuracy:", str(np.round(confidence_score * 100))[:-2], "%")
    time.sleep(5)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break


camera.release()
cv2.destroyAllWindows()
