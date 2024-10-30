import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(model, img_path, img_height=150, img_width=150, class_names=None):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    for class_name, prob in zip(class_names, predictions[0]):
        print(f"{class_name}: {prob*100:.2f}%")

    # Mostrar la clase con mayor probabilidad
    max_class = np.argmax(predictions)
    print(f"\nPredicción: {class_names[max_class]} ({predictions[0][max_class]*100:.2f}%)")
