from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(img_height=150, img_width=150, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_data = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, test_data
