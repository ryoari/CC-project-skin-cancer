from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import tensorflow as tf  
from tensorflow.keras.applications import VGG16  
from tensorflow.keras.layers import Dense, Flatten  
from tensorflow.keras.models import Model  

img_width, img_height = 224, 224


train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    validation_split=0.2  
)


train_data_dir = r"C:\Users\ryoar\Downloads\archive (1)"


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,  
    class_mode='binary',  
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,  
    class_mode='binary',  
    subset='validation'
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  


model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


epochs = 10  
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

model.save('skin_cancer_model.h5')  