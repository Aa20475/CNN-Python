
import matplotlib.pyplot as plt
import matplotlib.image as mi
import nn
import pickle




img = mi.imread(r'''C:\Users\aa204\Desktop\cnngit\p.jpg''')

print(img.shape)

plt.imshow(img)

plt.show()

cl = [[5,5,3],[2,2]]
p = [[3,2],[2,2]]

cnn = nn.make_cnn(cl)

n,par,data = nn.make_nn_arch(img,cnn,p)

Y = nn.forward_prop(n,par,data)

print(Y)
