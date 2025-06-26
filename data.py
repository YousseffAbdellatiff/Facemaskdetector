from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_path="dataset", img_size=224, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_data = datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',   # <-- change this!
        subset='training',
        shuffle=True
    )
    val_data = datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',   # <-- change this!
        subset='validation',
        shuffle=True
    )
    return train_data, val_data