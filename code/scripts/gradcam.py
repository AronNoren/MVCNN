import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
import matplotlib.pyplot as plt
from utils.explainability import *
from models.MVCNN import get_model
from utils.dataloader import get_data
""" python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
Makes the visualization. """

classes = ['Airplane','Bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',
'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',
'laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink',
'sofa','stairs','stool','table','tent','toiet','tv_stand','vase','wardrobe','xbox']

model = get_model()
PATH = 'code/models/saved_models/mvcnn_stage_fine.pkl'
model.load_state_dict(torch.load(PATH))
#print(model)
grad_cam = GradCam(model=model, feature_module=model.features, \
                       target_layer_names=["7"], use_cuda=False)
data_loaders = get_data()
test = data_loaders['val']
print(test)
for batch in test:
    labels,inputs = batch[0],batch[1]
    #print(labels)
    inputs = inputs.to('cpu')
    labels = labels.to('cpu')
    break 
img = inputs[0,:,:,:,:]
print("done with part 1")
#plt.imshow(torch.transpose(img[0,:,:,:],0,2))
#plt.show()
print(img.shape)
pred = model(inputs)
e_x = torch.exp(pred - torch.max(pred))
pred = e_x / torch.sum(e_x)
conf = torch.max(pred)
clas = torch.argmax(pred)
#img = cv2.imread(args.image_path, 1)
#img = np.float32(img) / 255
# Opencv loads as BGR:
#img = img[:, :, ::-1]
#input_img = preprocess_image(img)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = None
grayscale_cam = grad_cam(img, target_category)
print("grayscale")
print(grayscale_cam.shape)
#grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
print(torch.max(img))
print(torch.min(img))
img = img - torch.min(img)
img = img/torch.max(img)
print(np.max(grayscale_cam))
print(np.min(grayscale_cam))
#cam = show_cam_on_image(torch.transpose(torch.transpose(img[0,:,:,:],0,2),0,1), grayscale_cam[0,:,:])


print('prediction: ' + classes[clas])
print('confidence: ' + str(conf))
print(torch.sum(pred))
images = []

for image in range(12):
    cam = show_cam_on_image(torch.transpose(torch.transpose(img[image,:,:,:],0,2),0,1), grayscale_cam[image,:,:])
    cv2.imshow(str(image), cam)
    #plt.imshow(torch.transpose(img[image,:,:,:],0,2))
    cv2.waitKey(0)  
    images.append(cam)
row1 = cv2.hconcat((images[0], images[1],images[2],images[3],images[4],images[5]))
row2 = cv2.hconcat((images[6], images[7],images[8],images[9],images[10],images[11]))
final = cv2.vconcat((row1,row2))
cv2.imwrite("cam.jpg", final)
#plt.imshow(grayscale_cam)
#plt.show()
gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
gb = gb_model(input_img, target_category=target_category)
gb = gb.transpose((1, 2, 0))

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask*gb)
gb = deprocess_image(gb)

cv2.imwrite("cam.jpg", cam)
cv2.imwrite('gb.jpg', gb)
cv2.imwrite('cam_gb.jpg', cam_gb)