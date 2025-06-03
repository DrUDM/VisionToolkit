# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import time 
  
import itertools 
import torch
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt 

from vision_toolkit.segmentation.processing.ternary_segmentation import TernarySegmentation as Segmentation 
from vision_toolkit.segmentation.segmentation_algorithms.I_CNN import CNN1D
from vision_toolkit.segmentation.utils.segmentation_utils import dict_vectorize, standard_normalization


class DLTraining():
        
    def __init__(self, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task, config_df = None,
                 **kwargs):
        
        kwargs.update({'distance_type': 'angular'})
     
        self.meta_config = dict({
            'sampling_frequency': sampling_frequency,
            'segmentation_method': segmentation_method,
            'task': task,
            'distance_projection': kwargs.get('distance_projection'),  
            'distance_type': kwargs.get('distance_type'), 
            'neural_network': kwargs.get('neural_network', 'cnn'),
            'size_plan_x': kwargs.get('size_plan_x'),
            'size_plan_y': kwargs.get('size_plan_y'),
            'smoothing': kwargs.get('smoothing', 'savgol'),
            'min_int_size': kwargs.get('min_int_size', 2),
            'display_results': kwargs.get('display_results', True),
            'display_segmentation': kwargs.get('display_segmentation', False),
            'verbose': kwargs.get('verbose', True),
            'feature_normalization': False, 
            'init_learning_rate': kwargs.get('init_learning_rate', 1e-3),
            'batch_size': kwargs.get('batch_size', 1024),
            'epochs': kwargs.get('epochs', 15),
                       })
       
        if task == 'binary':
            classes = 2
            
        elif task == 'ternary':
            classes = 3
          
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
       
        self.networks = dict({
            'cnn': CNN1D(numChannels = 3, classes = classes).to(device),            
                })
        
        individual_features = [
            Segmentation.generate_features(input_df, 
                                           sampling_frequency, segmentation_method, 
                                           **kwargs) for input_df in input_dfs
            ]
                
        feature_mat = np.array(list(itertools.chain.from_iterable([list(feat_v) 
                                                                   for feat_v in individual_features])))
    
        individual_labels = [
            pd.read_csv(event_df).to_numpy().flatten() for event_df in event_dfs
            ]
       
        labels = np.array(list(itertools.chain.from_iterable([list(labl) 
                                                              for labl in individual_labels])))
        t_k = labels>0
        
        self.feature_mat = feature_mat[t_k]
        self.labels = labels[t_k]  
            
        
    @classmethod
    def hard_fit(cls, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method,
                 **kwargs):
        
        mt = cls(input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method,
                 **kwargs)
        
        feature_mat = mt.feature_mat
            
        train_X, train_y = feature_mat, mt.labels
      
        # Convert to float32 for pytorch
        train_X = train_X.astype(np.float32)
        
        train_y -= 1
        #train_y = train_y.astype(int)
         
        # Initialize neural network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        model = mt.networks[mt.meta_config['neural_network']]
        
        # Get learning rate, batch size and number of epochs
        init_lr = mt.meta_config['init_learning_rate'] 
        b_s = mt.meta_config['batch_size']  
        e_ = mt.meta_config['epochs'] 
        
        # calculate steps per epoch for training set
        trainSteps = len(train_X) // b_s
        
        # initialize the optimizer and loss function
        opt = torch.optim.Adam(model.parameters(), lr=init_lr)
        lossFn = torch.nn.CrossEntropyLoss() 
        
        # initialize a dictionary to store training history
        H = {
        	"train_loss": [],
        	"train_acc": [], 
        }
        
        # measure how long training is going to take
        print("Training the network...")
        startTime = time.time()
        
        
        
        tensor_train_X = torch.tensor(train_X)
        tensor__train_y = torch.tensor(train_y)
         
        trainData = torch.utils.data.TensorDataset(tensor_train_X, 
                                                   tensor__train_y)
        trainDataLoader = torch.utils.data.DataLoader(trainData, 
                                                      shuffle=True,
                                                      batch_size=b_s)
         
        # loop over epochs
        for e in range(e_):
            
        	# set the model in training mode
        	model.train()
        	
            # initialize the total training loss
        	totalTrainLoss = 0 
        	
            # initialize the number of correct predictions in the training step
        	trainCorrect = 0 
        	
            # loop over the training set
        	for (x, y) in trainDataLoader:
                
        		# send the input to the device
        		(x, y) = (x.to(device), y.to(device))
        		
                # perform a forward pass and calculate the training loss
        		pred = model(x) 
        		loss = lossFn(pred, y)
        		
                # zero out the gradients, perform the backpropagation step,
        		# and update the weights
        		opt.zero_grad()
        		loss.backward()
        		opt.step()
        		
                # add the loss to the total training loss so far and
        		# calculate the number of correct predictions
        		totalTrainLoss += loss
        		trainCorrect += (pred.argmax(1) == y).type(
        			torch.float).sum().item()
               
        	# calculate the average training loss
        	avgTrainLoss = totalTrainLoss / trainSteps
         
        	# calculate the training accuracy
        	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
         
        	# update the training history
        	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        	H["train_acc"].append(trainCorrect)  
        	
            # print the model training information
        	print("Epoch: {}/{}".format(e + 1, e_))
        	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        		avgTrainLoss, trainCorrect))
             
        # finish measuring how long training took
        endTime = time.time()
        print("Total time taken to train the model: {:.2f}s".format(
        	endTime - startTime))
   
        path = 'segmentation_src/segmentation_algorithms/trained_models/{sm}/i_{net}_{task}.pt'.format(task = mt.meta_config['task'],
                                                                                                       sm = segmentation_method,
                                                                                                       net = mt.meta_config['neural_network'])
        
        torch.save(model.state_dict(), path)
     
  
         
         
    @classmethod
    def fit_predict(cls, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method,
                 **kwargs):
        
        mt = cls(input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method,
                 **kwargs)
        
        feature_mat = mt.feature_mat
          
        train_X, test_X, train_y, test_y = train_test_split(feature_mat, 
                                                            mt.labels, 
                                                            test_size = 0.25,
                                                            shuffle=False)
 
        # Convert to float32 for pytorch
        train_X = train_X.astype(np.float32)
        
        train_y -= 1
        #train_y = train_y.astype(int)
        
        # Convert to float32 for pytorch
        test_X = test_X.astype(np.float32)
        
        test_y -= 1
        #test_y = test_y.astype(int)
        
        # Initialize neural network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        model = mt.networks[mt.meta_config['neural_network']]
        
        # Get learning rate, batch size and number of epochs
        init_lr = mt.meta_config['init_learning_rate'] 
        b_s = mt.meta_config['batch_size']  
        e_ = mt.meta_config['epochs'] 
        
        # calculate steps per epoch for training set
        trainSteps = len(train_X) // b_s
        
        # initialize the optimizer and loss function
        opt = torch.optim.Adam(model.parameters(), lr=init_lr)
        lossFn = torch.nn.NLLLoss() 
        
        # initialize a dictionary to store training history
        H = {
        	"train_loss": [],
        	"train_acc": [], 
        }
        
        # measure how long training is going to take
        print("Training the network...")
        startTime = time.time()
         
        tensor_train_X = torch.tensor(train_X)
        tensor__train_y = torch.tensor(train_y)
        
        tensor_test_X = torch.tensor(test_X)
        tensor_test_y = torch.tensor(test_y)
         
        trainData = torch.utils.data.TensorDataset(tensor_train_X, 
                                                   tensor__train_y)
        trainDataLoader = torch.utils.data.DataLoader(trainData, 
                                                      shuffle=True,
                                                      batch_size=b_s)
        
        testData = torch.utils.data.TensorDataset(tensor_test_X, 
                                                  tensor_test_y)
        testDataLoader = torch.utils.data.DataLoader(testData, 
                                                     batch_size=b_s)
        
        # loop over epochs
        for e in range(e_):
            
        	# set the model in training mode
        	model.train()
        	
            # initialize the total training loss
        	totalTrainLoss = 0 
        	
            # initialize the number of correct predictions in the training step
        	trainCorrect = 0 
        	
            # loop over the training set
        	for (x, y) in trainDataLoader:
                
        		# send the input to the device
        		(x, y) = (x.to(device), y.to(device))
        		
                # perform a forward pass and calculate the training loss
        		pred = model(x) 
        		loss = lossFn(pred, y)
        		
                # zero out the gradients, perform the backpropagation step,
        		# and update the weights
        		opt.zero_grad()
        		loss.backward()
        		opt.step()
        		
                # add the loss to the total training loss so far and
        		# calculate the number of correct predictions
        		totalTrainLoss += loss
        		trainCorrect += (pred.argmax(1) == y).type(
        			torch.float).sum().item()
               
        	# calculate the average training loss
        	avgTrainLoss = totalTrainLoss / trainSteps
         
        	# calculate the training accuracy
        	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
         
        	# update the training history
        	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        	H["train_acc"].append(trainCorrect)  
        	
            # print the model training information
        	print("Epoch: {}/{}".format(e + 1, e_))
        	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        		avgTrainLoss, trainCorrect))
             
        # finish measuring how long training took
        endTime = time.time()
        print("Total time taken to train the model: {:.2f}s".format(
        	endTime - startTime))

        # Evaluation on the test set
        print("Evaluating network...")
        
        # turn off autograd for testing evaluation
        with torch.no_grad():
            
        	# set the model in evaluation mode
        	model.eval()
        	
        	# initialize a list to store predictions
        	preds = []
        	
            # loop over the test set
        	for (x, y) in testDataLoader:
        		
                # send the input to the device
        		x = x.to(device)
        		
                # make the prediction
        		pred = model(x)
        		preds.extend(pred.argmax(axis=1).cpu().numpy())
        
        # generate a classification report
        if mt.meta_config['task'] == 'binary':
            clss = np.array(['fix', 'sac'])
            
        elif mt.meta_config['task'] == 'ternary':
            clss = np.array(['fix', 'sac', 'sp'])
            
        print(classification_report(test_y,
        	np.array(preds), target_names=clss))
                 
        
        
class MLTraining():
    
    def __init__(self, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs):
        
        if segmentation_method == 'I_HOV':
            kwargs.update({'distance_type': 'euclidean'})
            
        else:
            kwargs.update({'distance_type': 'angular'})
        
        self.meta_config = dict({
            'sampling_frequency': sampling_frequency,
            'segmentation_method': segmentation_method,
            'task': task,
            'distance_projection': kwargs.get('distance_projection'),  
            'distance_type': kwargs.get('distance_type'), 
            'classifier': kwargs.get('classifier', 'rf'),
            'size_plan_x': kwargs.get('size_plan_x'),
            'size_plan_y': kwargs.get('size_plan_y'),
            'smoothing': kwargs.get('smoothing', 'savgol'),
            'min_int_size': kwargs.get('min_int_size', 2),
            'display_results': kwargs.get('display_results', True),
            'display_segmentation': kwargs.get('display_segmentation', False),
            'verbose': kwargs.get('verbose', True)
                       })
      
        # Features already normalized as distributions during feature extraction
        if segmentation_method == 'I_HOV':
            
            self.meta_config.update({
                'feature_normalization': False,  
                #'distance_type': 'euclidean',
                    }) 
            
        # Normalization required while using SVM or KNN classifier
        elif segmentation_method == 'I_FC':
             
            if self.meta_config['classifier'] == 'rf': 
                self.meta_config.update({
                    'feature_normalization': False,  
                        }) 
                
            else:
                self.meta_config.update({
                    'feature_normalization': True,  
                        }) 
           
        self.classifiers = dict({
            'rf': RandomForestClassifier(max_depth=10, max_features='sqrt'),
            'svm': SVC(), 
            'knn': KNeighborsClassifier(n_neighbors=3)                 
                })
        
        individual_features = [
            Segmentation.generate_features(input_df, 
                                           sampling_frequency, segmentation_method, 
                                           **kwargs) for input_df in input_dfs
            ]
        
        if segmentation_method == 'I_FC':
            feature_mat = dict_vectorize(individual_features)
            
        elif segmentation_method == 'I_HOV':
            feature_mat = np.array(list(itertools.chain.from_iterable([list(feat_v) 
                                                                       for feat_v in individual_features])))
           
        individual_labels = [
            pd.read_csv(event_df).to_numpy().flatten() for event_df in event_dfs
            ]
       
        labels = np.array(list(itertools.chain.from_iterable([list(labl) 
                                                              for labl in individual_labels])))
        t_k = labels>0
        
        self.feature_mat = feature_mat[t_k]
        self.labels = labels[t_k]
        
  
    @classmethod
    def hard_fit(cls, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs):
     
        mt = cls(input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs)
       
        feature_mat = mt.feature_mat
        
        if mt.meta_config['feature_normalization']:
            feature_mat, mu, sigma = standard_normalization(feature_mat)
            print('\n --- Features normalized ---\n')
          
            norm_ = np.array([mu, sigma])
            
            path = 'segmentation_src/segmentation_algorithms/trained_models/{sm}/normalization_{task}.npy'.format(
                sm = segmentation_method,
                task = mt.meta_config['task']
                )
            
            np.save(path, norm_)
            
        train_features, train_labels, = feature_mat, mt.labels 
        
        clf = mt.classifiers[mt.meta_config['classifier']]
        clf.fit(train_features, train_labels)
         
        path = path = 'segmentation_src/segmentation_algorithms/trained_models/{sm}/i_{cl}_{task}.joblib'.format(
            sm = segmentation_method,
            cl = mt.meta_config['classifier'],
            task = mt.meta_config['task']
            )
        
        joblib.dump(clf, path)
        
             
    @classmethod
    def fit_predict(cls, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs):
        
        mt = cls(input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs)
       
        feature_mat = mt.feature_mat
        
        if mt.meta_config['feature_normalization']:
            feature_mat, mu, sigma = standard_normalization(feature_mat)
            print('\n --- Features normalized ---\n')
        print(feature_mat.shape) 
        train_features, test_features, train_labels, test_labels = train_test_split(feature_mat, 
                                                                                    mt.labels, 
                                                                                    test_size = 0.25,
                                                                                    shuffle=False)
        
        clf = mt.classifiers[mt.meta_config['classifier']]
        clf.fit(train_features, train_labels)
        
        pred = clf.predict(test_features)
        
        if mt.meta_config['verbose']:
            
            print('\n --- Config used: ---\n')
            
            for it in mt.meta_config.keys():
                print('# {it}:{esp}{val}'.format(it=it,
                                                esp = ' '*(30-len(it)),
                                                val = mt.meta_config[it]))
            print('\n')
            
        if mt.meta_config['task'] == 'binary':
            clss = np.array(['fix', 'sac'])
            
        elif mt.meta_config['task'] == 'ternary':
            clss = np.array(['fix', 'sac', 'sp'])
            
        print(classification_report(test_labels,
                                    pred, 
                                    target_names= clss))
    
  
 


 
        