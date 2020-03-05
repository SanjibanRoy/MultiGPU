import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

with tf.device('/cpu:0'):
   model = Xception(weights=None,
      input_shape=(height, width, 3),
         classes=num_classes)

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop')

x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))


parallel_model.fit(x, y, epochs=20, batch_size=256)

   
model.save('my_model.h5')

# Not needed to change the device scope for model definition:
model = Xception(weights=None)

try:
    model = multi_gpu_model(model, cpu_relocation=False)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")

#model.compile()

