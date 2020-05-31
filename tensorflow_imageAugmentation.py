from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import matplotlib.pyplot as plt
KNOWN_FACES_DIR = "known_faces"


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = load_img(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		#import cv2
		x = img_to_array(image)
		x = x.reshape((1,) + x.shape)
		
		i = 0
		for batch in datagen.flow(x , batch_size=1 , save_to_dir = f"{KNOWN_FACES_DIR}/{name}" , save_prefix=filename , save_format = 'PNG') :
			i+=1
			if i > 1 :
				break








#img = load_img('pic1.PNG')  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir=os.listdir(f"{KNOWN_FACES_DIR}/{name}"), save_prefix='atharva', save_format='PNG'):
#    i += 1
#    if i > 20:
#        break