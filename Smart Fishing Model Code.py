import warnings
warnings.filterwarnings('ignore')
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input



IMAGE_SIZE = [224, 224]

train_path = r'C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish Train'
test_path = r'C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish Test'
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
folders = glob(r'C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish Train\*')
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




training_set = train_datagen.flow_from_directory(r'C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish Train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory(r'C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish Test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model.save('Smart Fishing Model.h5')
model=load_model('Smart Fishing Model.h5')
img=image.load_img(r"C:\Users\Moldu\Data preprocesing\Data Preprocessing\Data\Fish Data\Fish\1.jpg",target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
result=int(classes[0][0])
if result==0:
    print("Shark")
else:
    print("Fish")







