
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mi
import pickle
#NN

def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)

def sigmoid_back(dA,Z):
    sig = sigmoid(z)
    return dA*(sig)*(1-sig)

def relu_back(dA,Z):
    dZ = np.array(dA,copy=True)
    dZ[Z<=0] = 0
    return dZ 



def make_nn(p):
    nn_architecture = []
    for i in range(len(p)):
        if len(p[i])==2:
            nn_architecture.append({"input_dim": p[i][0], "output_dim": p[i][1], "activation": "relu"})
        else:
            print("ERROR in size at :",i+1)
    return nn_architecture

def init_layers(nn_architecture,seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}
       
    layer_idx = 0
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
       
        params_values['W' + str(layer_idx)] = np.random.uniform(low=-1.0, high=1.0, size=(layer_input_size, layer_output_size))
        params_values['b' + str(layer_idx)] = np.random.uniform(low=-1.0, high=1.0, size=(1,layer_output_size))
    return params_values

def sl_forward_prop(a_prev,w_curr,b_curr,activation="relu"):
    z_curr = np.dot(a_prev,w_curr) + b_curr
    if activation is "relu":
        act  = relu
    elif activation is "sigmoid":
        act = sigmoid
    else:
        act = relu

    return act(z_curr),z_curr

def forward_prop(nn_architecture,params,X):
    a_curr = X
    for idx , layer in enumerate(nn_architecture):
        layer_idx = idx
        a_prev = a_curr
        act = layer["activation"]
        w_curr = params["W"+str(layer_idx)]
        b_curr = params["b"+str(layer_idx)]

        a_curr ,z_curr = sl_forward_prop(a_prev,w_curr,b_curr,act)

    return a_curr

#CNN

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def convolve(img,filt):
    f_h,f_w = filt.shape
    i_h,i_w,k = img.shape
    
    filt = np.array(filt)
    
    mod = []
    for x in range(i_h-1-f_h):
        m = []
        for y in range(i_w-1-f_w):
            h=[]
            for z in range(k):
                p = 0
                p+=np.sum(np.multiply(img[x:x+f_h,y:y+f_w,z],filt))
                h.append(p)
            m.append(h)
        mod.append(m)
    mod = np.array(mod)
    return np.array(mod)



def max_pool(img,fx,fy):
    f_h,f_w = fy,fx
    i_h,i_w,k = img.shape

    mod = []
    for x in range(i_h-1-f_h):
        m = []
        for y in range(i_w-1-f_w):
            h=[]
            for z in range(k):
                h.append(np.amax(img[x:x+f_h,y:y+f_w,z]))
            m.append(h)
        mod.append(m)
    mod = np.array(mod)
    return np.array(mod)



def init_filter(size):
    return np.random.standard_normal(size=size) *0.01


def make_cnn(p):
    cnn = []
    for i in p:
        if len(i)==3:

            layer =[]
            for x in range(i[2]):
                layer.append(init_filter([i[0],i[1]]))
            cnn.append(["convolve",layer])
        elif len(i)==2:
            cnn.append(["pool",i[0],i[1]])
        else:
            print("Bad layer sizes")
    return cnn

def cnn_forward_prop(dat,cnn,img = False):
    if not img:
        data= dat
    else:
        data = [dat]
    for i in cnn:
        if i[0]=="pool":
            for p in data:
                p = max_pool(p,i[1],i[2])
                plt.imshow(p)

                plt.show()

        elif i[0]=="convolve":
            mod = []
            for j in i[1]:
                for k in data:
                    
                    m = convolve(k,j)
                    mod.append(m)
                    plt.imshow(m)   

                    plt.show()

            data = mod

    data =np.array(data).flatten()
    return data

def make_nn_arch(img,cnn,p):
    data = cnn_forward_prop(img,cnn,img= True)
    x =  data.shape
    p.insert(0,[x[0],p[0][0]])
    print(p)
    nn = make_nn(p)
    params = init_layers(nn)

    return nn,params,data


def full_forward_prop(data,cnn,nn,params,img = False):
    
    if not img:
        data = cnn_forward_prop(data,c,img=False)
    else:
        data = cnn_forward_prop(data,c,img=True)

    Y = forward_prop(nn,params,data)

    return Y






    
    
    
