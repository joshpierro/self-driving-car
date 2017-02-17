from keras.models import load_model
import matplotlib.image as mpimg
import numpy as np
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = load_model('/Users/pierroj/josh/sdc/p3/model.h5')
print(model.inputs)
print(model.summary())

x = []
image_path = img_center = 'firstimage.jpg'
image = mpimg.imread(img_center)



x.append(image)
x = np.array(x)

predict = model.predict(x,batch_size=1,verbose=1)
print(predict)

print(np.argmax(predict))

#predict2 = model.predict(x)
#print(float(predict.item(0)))



#print('Predicted:', decode_predictions(predict2)[0])