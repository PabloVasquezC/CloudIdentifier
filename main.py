from model import create_model
from data_loader import load_data
from utils import predict_image
import matplotlib.pyplot as plt

# Configuraciones generales
img_height, img_width = 150, 150
batch_size = 32
epochs = 50

if __name__ == "__main__":
    # Cargar datos
    train_data, test_data = load_data(img_height, img_width, batch_size)

    # Crear y entrenar el modelo
    model = create_model(img_height, img_width, num_classes=train_data.num_classes)
    history = model.fit(train_data, epochs=epochs, validation_data=test_data)

    # Evaluar modelo
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'Accuracy en el conjunto de test: {test_accuracy*100:.2f}%')

    # Graficar precisión y pérdida
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

    # Predecir en una imagen de ejemplo
    predict_image(
        model,
        'Data/test/Cirrus/1ec32b25-bbd8-40f7-a97d-f827414df2f2.jpg', 
        img_height,
        img_width,
        class_names=list(train_data.class_indices.keys())
    )
