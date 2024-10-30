import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(model, img_path, img_height=150, img_width=150, class_names=None):
    # Cargar y preparar la imagen para el modelo
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para batch

    # Realizar predicción
    predictions = model.predict(img_array)

    # Obtener las probabilidades y la clase con mayor probabilidad
    class_probabilities = predictions[0]
    max_class_index = np.argmax(class_probabilities)
    max_class_name = class_names[max_class_index]
    max_probability = class_probabilities[max_class_index] * 100

    # Imprimir los resultados
    print("Probabilidades de cada tipo de nube:")
    for class_name, prob in zip(class_names, class_probabilities):
        print(f"{class_name}: {prob*100:.2f}%")

    print(f"\nPredicción más probable: {max_class_name} con {max_probability:.2f}% de confianza")
