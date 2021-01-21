import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import os
import cv2
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cart_path = os.path.dirname(os.path.realpath(__file__))+'\\data\\'
pol_path = os.path.dirname(os.path.realpath(__file__))+'\\sample_sonar_images\\'
#files = os.listdir(binfile_path)

#print(os.listdir('.'))
im_files = os.listdir(cart_path)
#print(im_files)
for nr in range(95,2449):
    cartname = 'Pnr_'+ str(nr)+'.jpg'
    polname = 'Ping_number_'+str(nr)+'.jpg'
    if cartname in im_files:
        cart = np.array(cv2.imread(cart_path+cartname)) #read image
        pol = np.array(cv2.imread(pol_path+polname))
        black = [0,0,0]
        cart = cv2.copyMakeBorder(cart,58,58,58,58,cv2.BORDER_CONSTANT,value=black )
        frame = cv2.hconcat((cart, pol))
        #edge = np.ones((600-484, 123, 3))*255
        #print(cart)
        #cart = np.concatenate((cart,edge),axis = 0)
        #print(cart)
        #print(pol.shape)
        cv2.namedWindow('sonar')        # Create a named window
        cv2.moveWindow('sonar', 450,100)  # Move it to (40,30)
        cv2.imshow('sonar', frame)

        cv2.waitKey(10)
   
print(cart.shape)
cv2.destroyAllWindows()

'''
def animate(img_path):
    tstart = time.time()                   # for profiling
    data=np.random.randn(128,518)
    images = os.listdir(img_path)
    im=plt.imshow(data)

    for file in images:
        image= cv2.imread(file)
        #data=np.random.randn(10,10)
        im.set_data(image)
        fig.canvas.draw()                         # redraw the canvas

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate(binfile_path))
plt.show()
'''