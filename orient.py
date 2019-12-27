import numpy as np
from collections import Counter
import time
import math
import pickle
import sys
import pandas as pd

# KNN functions

def parse_image(filename):
    with open(filename, "r") as f:
        image_class = []
        image_data = []
        image_output = []
        images = [line for line in f.read().split("\n") if line != '']
        for i in images:
            image_temp = i.split()
            image_output.append(image_temp[0])
            image_class.append(int(image_temp[1]))
            image_data.append(list(map(int, image_temp[2:len(image_temp)])))
    return np.array(image_class), np.array(image_data), image_output

# Decision tree functions

def calculate_entropy(rows,labels, k):
        
        percentile_list = np.percentile(rows, [20,40,60,70])
       
        min_entropy = 99999
        min_alpha = 0
        for alpha in percentile_list:
        
            temp_Dict = {}
            numTotalObs = len(rows)      

            #Step 1- Split the Rows
            pos_Rows = rows[rows >= alpha]
            neg_Rows = rows[rows < alpha]

            #Step 2- Split the labels
            pos_Labels = labels[rows >= alpha]
            neg_Labels = labels[rows < alpha]

            temp_Dict['left'] = dict(Counter(pos_Labels))
            temp_Dict['right'] = dict(Counter(neg_Labels))

            #Calculate the probabilities and compute their entropy
            pos_Entropy = 0
            for Labelcount in temp_Dict['left'].values():
                pos = Labelcount/len(pos_Rows)
                pos_Entropy+=((-pos)*math.log2(pos))

            neg_Entropy = 0
            for Labelcount in temp_Dict['right'].values():
                neg = Labelcount/len(neg_Rows)
                neg_Entropy+=((-pos)*math.log2(pos))

            final_Entropy = (len(pos_Rows)/numTotalObs)*pos_Entropy + (len(neg_Rows)/numTotalObs)*neg_Entropy 
            if min_entropy > final_Entropy:
                min_entropy = final_Entropy
                min_alpha = alpha
            
        return min_entropy,k, min_alpha
    
    
def find_split(train_data, train_image_class):
    entropy_values = []
    for i in range(0,len(train_data[0])):
        entropy_values.append(calculate_entropy(train_data[:,i], train_image_class,i))
    entropy, split_column, split_value = min(entropy_values)
    return split_column, split_value

# Neural network functions

def relu(p):
    return np.maximum(0, p)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu_backprop(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def sigmoid_backprop(x):
    return sigmoid(x)*(1- sigmoid(x))

def softmax(u):
    return np.exp(u) / np.sum(np.exp(u), axis=0, keepdims=True)

def initialize_parameters(dims):
    for i in range(1, len(dims)):
        parameters["W" + str(i)] = np.random.randn(dims[i], dims[i - 1]) * (np.sqrt(2 / dims[i - 1]))
        parameters["b" + str(i)] = np.zeros((dims[i], 1))
    return parameters

def forward_prop(parameters, X_train, activation):
    
    Z["Z1"] = np.dot(parameters["W1"], X_train) + parameters["b1"]
    activation["A1"] = relu(Z["Z1"])
    
    dim = len(dimensions)
    
    for i in range(2, dim-1):
        Z["Z" + str(i)] = np.dot(parameters["W" + str(i)], activation["A" + str(i - 1)]) + parameters["b" + str(i)]
        activation["A" + str(i)] = relu(Z["Z" + str(i)])
        
    Z["Z" + str(dim-1)] = np.dot(parameters["W" + str(dim-1)], activation["A" + str(dim-2)]) + parameters["b" + str(dim-1)]
    activation["A" + str(dim-1)] = softmax(Z["Z" + str(dim-1)])
    
    return Z, activation

def compute_cost(activation):
    loss = - np.sum((Y_train * np.log(activation["A" + str(len(dimensions)-1)])), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / X_train.shape[1]
    return cost

def gradient_loss_calculation(parameters, Z, activation):
    grad_loss["dZ" + str(len(dimensions)-1)] = (activation["A" + str(len(dimensions)-1)] - Y_train) / X_train.shape[1]
    dim = len(dimensions)-1
    
    for i in range(1, dim):
        grad_loss["dA" + str(dim - i)] = np.dot(parameters["W" + str(dim - i + 1)].T, grad_loss["dZ" + str(dim - i + 1)])
        grad_loss["dZ" + str(dim - i)] = grad_loss["dA" + str(dim - i)] * relu_backprop(Z["Z" + str(dim - i)])
        
    grad_loss["dW1"] = np.dot(grad_loss["dZ1"], X_train.T)
    grad_loss["db1"] = np.sum(grad_loss["dZ1"], axis=1, keepdims=True)
    
    for i in range(2, dim+1):
        grad_loss["dW" + str(i)] = np.dot(grad_loss["dZ" + str(i)], activation["A" + str(i - 1)].T)
        grad_loss["db" + str(i)] = np.sum(grad_loss["dZ" + str(i)], axis=1, keepdims=True)
        
    return parameters, Z, activation, grad_loss

def back_prop(grad_loss, learning_rate=0.05):
    for i in range(1, len(dimensions)):
        parameters["W" + str(i)] = parameters["W" + str(i)] - (learning_rate * grad_loss["dW" + str(i)])
        parameters["b" + str(i)] = parameters["b" + str(i)] - (learning_rate * grad_loss["db" + str(i)])
    return parameters

def neural_net(num_iterations, activation):
    parameters = initialize_parameters(dimensions)
    for i in range(0, num_epochs):
        Z, activation = forward_prop(parameters, X_train, activation)
        cost = compute_cost(activation)
        parameters, Z, activation, grad_loss = gradient_loss_calculation(parameters, Z, activation)
        parameters = back_prop(grad_loss, learning_rate=0.05)
        
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters

def predict(parameters, X_test):
    forward_prop(parameters, X_test, activation)
    predictions = activation["A" + str(len(dimensions)-1)]
    return predictions



# Main Function
if __name__ == "__main__":
    train_test = sys.argv[1]
    filename = sys.argv[2]
    load_model = sys.argv[3]
    model_name = sys.argv[4]
    
    if train_test == 'train' and model_name == 'nearest':
        train_image_class, train_image_data, train_image_output = parse_image(filename)
        PIK = "nearest_model.dat"
        with open(PIK, "wb") as f:
            pickle.dump((train_image_data, train_image_class), f)
            
    if train_test == 'test' and model_name == 'nearest':
        # Reading
        with open(load_model, "rb") as f:
            train_image_data,train_image_class  = pickle.load(f)
            
        test_image_class, test_image_data, test_image_output = parse_image(filename)
        
        final_predicted = []
        k=0
        flag=0
        for i in test_image_data:
            intermediate = train_image_data - i
            #Euclidean distance
            int1 = np.sqrt(np.sum(np.square(intermediate), axis = 1))
            counts = Counter(train_image_class[list(int1.argsort()[::-1][-21:])])
            
            with open('output.txt', 'a') as f:
                x = sorted(counts.items() , key = lambda x: x[1], reverse = True)[0][0]
                if f.tell() == 0:
                    f.write(str(test_image_output[k])+ " "+str(x)+'\n')
                    k = k +  1
                    flag=1
                elif flag==0:
                    f.close()
                    with open('output.txt', 'w') as file:
                        file.write(str(test_image_output[k])+ " "+str(x)+'\n')
                        k = k +  1
                        flag=1
                else:
                    f.write(str(test_image_output[k])+ " "+str(x)+'\n')
                    k = k +  1
            
            final_predicted.append(sorted(counts.items() , key = lambda x: x[1], reverse = True)[0][0])
            
        print(np.mean(final_predicted == test_image_class))
        
    if train_test == 'train' and model_name == 'tree':
        train_image_class, train_data, train_image_output = parse_image(filename)
        
        h = []
        treequeue = []
        h.append(0)
        
        depth=255
        split_column, split_value = find_split(train_data, train_image_class)
        train_data_left = train_data[train_data[:,int(split_column)] < split_value]
        train_data_right = train_data[train_data[:,int(split_column)] >= split_value]
        h.append((split_column, split_value))
        split_left_class_label = train_image_class[train_data[:,int(split_column)] < split_value]
        split_right_class_label = train_image_class[train_data[:,int(split_column)] >= split_value]
        treequeue.append((train_data_left,split_left_class_label))
        treequeue.append((train_data_right,split_right_class_label))
        while depth>0 and len(treequeue)>0:
            train_data, train_image_class  = treequeue.pop(0)
            split_column, split_value = find_split(train_data,train_image_class)
            train_data_left = train_data[train_data[:,int(split_column)] < split_value]
            split_left_class_label = train_image_class[train_data[:,int(split_column)] < split_value]
            train_data_right = train_data[train_data[:,int(split_column)] >= split_value]
            split_right_class_label = train_image_class[train_data[:,int(split_column)] >= split_value]
            h.append((split_column, split_value))
            treequeue.append((train_data_left,split_left_class_label))
            treequeue.append((train_data_right,split_right_class_label))
            depth -= 1
        
        for i in treequeue:
            h.append((i[1],0))
            
        PIK = "tree_model.dat"

        with open(PIK, "wb") as f:
            pickle.dump(h, f)
            
    if train_test == 'test' and model_name == 'tree':
        test_image_class, test_image_data, test_image_output = parse_image(filename)
        
        # Reading
        with open(load_model, "rb") as f:
            h1 = pickle.load(f)
            
        final_predicted = []
        k=0
        flag=0
        for j in range(len(test_image_data)):
            i=1
            test = test_image_data[j,:]
            while i<len(h1)/2:
                split_column = h1[i][0]
                split_value = h1[i][1]
                if test[split_column] < split_value:
                    i = 2 * i
                else:
                    i = 2 * i + 1
            

            counts = Counter(h1[i][0])
            
            with open('output.txt', 'a') as f:
                x = sorted(counts.items() , key = lambda x: x[1], reverse = True)[0][0]
                if f.tell() == 0:
                    f.write(str(test_image_output[k])+ " "+str(x)+'\n')
                    k = k +  1
                    flag=1
                elif flag==0:
                    f.close()
                    with open('output.txt', 'w') as file:
                        file.write(str(test_image_output[k])+ " "+str(x)+'\n')
                        k = k +  1
                        flag=1
                else:
                    f.write(str(test_image_output[k])+ " "+str(x)+'\n')
                    k = k +  1
            final_predicted.append(sorted(counts.items() , key = lambda x: x[1], reverse = True)[0][0])
        

        print(np.mean(final_predicted==test_image_class))
        
        
    if train_test == 'train' and (model_name == 'nnet' or model_name == 'best'):
        train_image_class, train_image_data, train_image_output = parse_image(filename)
        
        Y_train = np.array(pd.get_dummies(train_image_class)).T
        
        X_train = train_image_data.T/ 255.
        
        parameters = {}
        Z = {}
        activation = {}
        grad_loss = {}
        dimensions = [X_train.shape[0], 128, 128, 4]
        
        
        num_epochs = 1000
        
        parameters = neural_net(num_epochs, activation)
        
        if model_name == 'nnet':
            PIK = "nnet_model.dat"
        else:
            PIK = "best_model.dat"

        with open(PIK, "wb") as f:
            pickle.dump((parameters, activation, dimensions, Z), f)
            
    if train_test == 'test' and (model_name == 'nnet' or model_name == 'best'):
        test_image_class, test_image_data, test_image_output= parse_image(filename)
        
        Y_test = np.array(pd.get_dummies(test_image_class)).T
        
        X_test = test_image_data.T/ 255.
        
        with open(load_model, "rb") as f:
            parameters,activation, dimensions, Z  = pickle.load(f)
            
        predictions = predict(parameters, X_test)
        max_prob = np.argmax(predictions.T, axis = 1)

        final_result = []
        k=0
        value_stored=0
        flag=0
        for i in max_prob:
            if i == 1:
                value_stored = 90
                final_result.append(90)
            elif i == 2:
                value_stored = 180
                final_result.append(180)
            elif i == 3:
                value_stored = 270
                final_result.append(270)
            else:
                value_stored = 0
                final_result.append(0)
            with open('output.txt', 'a') as f:
                if f.tell() == 0:
                    f.write(str(test_image_output[k])+ " "+str(value_stored)+'\n')
                    k = k +  1
                    flag=1
                elif flag==0:
                    f.close()
                    with open('output.txt', 'w') as file:
                        file.write(str(test_image_output[k])+ " "+str(value_stored)+'\n')
                        k = k +  1
                        flag=1
                else:
                    f.write(str(test_image_output[k])+ " "+str(value_stored)+'\n')
                    k = k +  1

        print(np.mean(final_result == test_image_class))
        