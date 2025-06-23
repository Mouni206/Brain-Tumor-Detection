import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def classify(image, model, class_names):
   
    image = image.resize((150, 150))  
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

 
    prediction = model.predict(image_array)[0]  

    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    return class_names[class_idx], confidence
