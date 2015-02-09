'''
Created on Jun 3, 2013

@author: sumanravuri
'''

import sys
import numpy as np
import scipy.io as sp
import scipy.sparse as ssp
import copy
import argparse
from Vector_Math import Vector_Math
import datetime
from scipy.special import expit
from RNNLM_Weight import RNNLM_Weight

class DSSM_Neural_Network_Language_Model(object, Vector_Math):
    """features are stored in format max_seq_len x nseq x nvis where n_max_obs is the maximum number of observations per sequence
    and nseq is the number of sequences
    weights are stored as nvis x nhid at feature level
    biases are stored as 1 x nhid
    rbm_type is either rbm_gaussian_bernoulli, rbm_bernoulli_bernoulli, logistic"""
    def __init__(self, config_dictionary): #completed
        """variables for Neural Network: feature_file_name(read from)
        required_variables - required variables for running system
        all_variables - all valid variables for each type"""
        self.feature_file_name = self.default_variable_define(config_dictionary, 'feature_file_name', arg_type='string')
        self.features, self.feature_sequence_lens = self.read_feature_file()
        self.model = RNNLM_Weight()
        self.output_name = self.default_variable_define(config_dictionary, 'output_name', arg_type='string')
        
        self.required_variables = dict()
        self.all_variables = dict()
        self.required_variables['train'] = ['mode', 'feature_file_name', 'output_name']
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'num_hiddens', 'weight_matrix_name', 
#                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 
                               'save_each_epoch',
                               'l2_regularization_const',
                               'steepest_learning_rate', 'momentum_rate',
                               'validation_feature_file_name', 'validation_label_file_name',
                               'use_maxent', 'nonlinearity',
                               'seed']
        self.required_variables['test'] =  ['mode', 'feature_file_name', 'weight_matrix_name', 'output_name']
        self.all_variables['test'] =  self.required_variables['test'] + ['label_file_name']
    
    def dump_config_vals(self):
        no_attr_key = list()
        print "********************************************************************************"
        print "Neural Network configuration is as follows:"
        
        for key in self.all_variables[self.mode]:
            if hasattr(self,key):
                print key, "=", eval('self.' + key)
            else:
                no_attr_key.append(key)
                
        print "********************************************************************************"
        print "Undefined keys are as follows:"
        for key in no_attr_key:
            print key, "not set"
        print "********************************************************************************"
    
    def default_variable_define(self,config_dictionary,config_key, arg_type='string', 
                                default_value=None, error_string=None, exit_if_no_default=True,
                                acceptable_values=None):
        #arg_type is either int, float, string, int_comma_string, float_comma_string, boolean
        try:
            if arg_type == 'int_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=True)
            elif arg_type == 'float_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=False)
            elif arg_type == 'int':
                return int(config_dictionary[config_key])
            elif arg_type == 'float':
                return float(config_dictionary[config_key])
            elif arg_type == 'string':
                return config_dictionary[config_key]
            elif arg_type == 'boolean':
                if config_dictionary[config_key] == 'False' or config_dictionary[config_key] == '0' or config_dictionary[config_key] == 'F':
                    return False
                elif config_dictionary[config_key] == 'True' or config_dictionary[config_key] == '1' or config_dictionary[config_key] == 'T':
                    return True
                else:
                    print config_dictionary[config_key], "is not valid for boolean type... Acceptable values are True, False, 1, 0, T, or F... Exiting now"
                    sys.exit()
            else:
                print arg_type, "is not a valid type, arg_type can be either int, float, string, int_comma_string, float_comma_string... exiting now"
                sys.exit()
        except KeyError:
            if error_string != None:
                print error_string
            else:
                print "No", config_key, "defined,",
            if default_value == None and exit_if_no_default:
                print "since", config_key, "must be defined... exiting now"
                sys.exit()
            else:
                if acceptable_values != None and (default_value not in acceptable_values):
                    print default_value, "is not an acceptable input, acceptable inputs are", acceptable_values, "... Exiting now"
                    sys.exit()
                if error_string == None:
                    print "setting", config_key, "to", default_value
                return default_value
    
    def read_feature_file(self, feature_file_name = None): #completed
        if feature_file_name is None:
            feature_file_name = self.feature_file_name
        try:
            feature_data = sp.loadmat(feature_file_name)
        except IOError:
            print "Unable to open ", feature_file_name, "... Exiting now"
            raise IOError
        
        if "num_rows" in feature_data:
            num_rows = feature_data['num_rows']
            num_cols = feature_data['num_cols']
            sparse_data = feature_data['data'].astype(np.int32).ravel()
            sparse_indices = feature_data['indices'].astype(np.int32).ravel()
            sparse_indptr = feature_data['indptr'].astype(np.int32).ravel()
            features = ssp.csr_matrix((sparse_data, sparse_indices, sparse_indptr), shape = (num_rows, num_cols))
            sequence_len = feature_data['feature_sequence_lengths']
            sequence_len = np.reshape(sequence_len, (sequence_len.size,))
        else:
            features, sequence_len = self.convert_to_sparse_mat(feature_data)
#            sequence_len = np.reshape(sequence_len, (sequence_len.size,))
        return features, sequence_len#in MATLAB format
    
    def convert_to_sparse_mat(self, feature_data):
        fsl = feature_data['feature_sequence_lengths'].ravel()
        num_obs = fsl.sum()
        data = np.ones((num_obs,))
        row = np.arange(num_obs)
        col = np.empty((num_obs,))
        cur_feat = 0
        for idx, feat_len in enumerate(fsl):
            end_feat = cur_feat + feat_len
            col[cur_feat:end_feat] = feature_data['features'][:feat_len,idx]
            cur_feat = end_feat
        num_col = int(np.max(col)) + 1
        print num_obs, num_col
        return ssp.csr_matrix((data,(row,col)), shape = (num_obs, num_col)), fsl
    
    def read_label_file(self, label_file_name = None): #completed
        """label file is a two-column file in the form
        sent_id label_1
        sent_id label_2
        ...
        """
        if label_file_name is None:
            label_file_name = self.label_file_name
        try:
            label_data = sp.loadmat(label_file_name)['labels'].astype(np.int32)
            return label_data#[:,1], label_data[:,0]#in MATLAB format
        except IOError:
            print "Unable to open ", label_file_name, "... Exiting now"
            sys.exit()
    
    def batch_size(self, feature_sequence_lens):
        return np.sum(feature_sequence_lens)
    
    def read_config_comma_string(self,input_string,needs_int=False):
        output_list = []
        for elem in input_string.split(','):
            if '*' in elem:
                elem_list = elem.split('*')
                if needs_int:
                    output_list.extend([int(elem_list[1])] * int(elem_list[0]))
                else:
                    output_list.extend([float(elem_list[1])] * int(elem_list[0]))
            else:
                if needs_int:
                    output_list.append(int(elem))
                else:
                    output_list.append(float(elem))
        return output_list

    def levenshtein_string_edit_distance(self, string1, string2): #completed
        dist = dict()
        string1_len = len(string1)
        string2_len = len(string2)
        
        for idx in range(-1,string1_len+1):
            dist[(idx, -1)] = idx + 1
        for idx in range(-1,string2_len+1):
            dist[(-1, idx)] = idx + 1
            
        for idx1 in range(string1_len):
            for idx2 in range(string2_len):
                if string1[idx1] == string2[idx2]:
                    cost = 0
                else:
                    cost = 1
                dist[(idx1,idx2)] = min(
                           dist[(idx1-1,idx2)] + 1, # deletion
                           dist[(idx1,idx2-1)] + 1, # insertion
                           dist[(idx1-1,idx2-1)] + cost, # substitution
                           )
                if idx1 and idx2 and string1[idx1]==string2[idx2-1] and string1[idx1-1] == string2[idx2]:
                    dist[(idx1,idx2)] = min (dist[(idx1,idx2)], dist[idx1-2,idx2-2] + cost) # transposition
        return dist[(string1_len-1, string2_len-1)]    

    def check_keys(self, config_dictionary): #completed
        print "Checking config keys...",
        exit_flag = False
        
        config_dictionary_keys = config_dictionary.keys()
        
        if self.mode == 'train':
            correct_mode = 'train'
            incorrect_mode = 'test'
        elif self.mode == 'test':
            correct_mode = 'test'
            incorrect_mode = 'train'
            
        for req_var in self.required_variables[correct_mode]:
            if req_var not in config_dictionary_keys:
                print req_var, "needs to be set for", correct_mode, "but is not."
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        for var in config_dictionary_keys:
            if var not in self.all_variables[correct_mode]:
                print var, "in the config file given is not a valid key for", correct_mode
                if var in self.all_variables[incorrect_mode]:
                    print "but", var, "is a valid key for", incorrect_mode, "so either the mode or key is incorrect"
                else:
                    string_distances = np.array([self.levenshtein_string_edit_distance(var, string2) for string2 in self.all_variables[correct_mode]])
                    print "perhaps you meant ***", self.all_variables[correct_mode][np.argmin(string_distances)], "\b*** (levenshtein string edit distance", np.min(string_distances), "\b) instead of ***", var, "\b***?"
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        if exit_flag:
            print "Exiting now"
            sys.exit()
        else:
            print "seems copacetic"

    def check_labels(self): #want to prune non-contiguous labels, might be expensive
        #TODO: check sentids to make sure seqences are good
        print "Checking labels..."
        if len(self.labels.shape) != 2 :
            print "labels need to be in (n_samples,2) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if self.labels.shape[0] != sum(self.feature_sequence_lens):
            print "Number of examples in feature file: ", sum(self.feature_sequence_lens), " does not equal size of label file, ", self.labels.shape[0], "... Exiting now"
            sys.exit()          
        print "labels seem copacetic"
    
#    def visible_to_hidden(self, inputs, weights):
#        num_obs = inputs.shape[0]
#        hiddens = np.zeros((num_obs, weights.shape[1]))
#        for obs_index in range(num_obs):
#            start_index = inputs.indptr[obs_index]
#            end_index = inputs.indptr[obs_index+1]
#            hiddens[obs_index] = weights[inputs.indices[start_index:end_index]].sum(axis=0)
#        
#        return hiddens
            
    def forward_pass_single_batch(self, inputs, model = None, return_hiddens = False):
        """forward pass for single batch size. Mainly for speed in this case
        """
        if model == None:
            model = self.model
        num_observations = inputs.shape[0]
        hiddens = inputs.dot(model.weights['visible_hidden']) #hack because of crappy sparse matrix support
#        hiddens = self.visible_to_hidden(inputs, model.weights['visible_hidden'])
        hiddens[:1,:] += self.weight_matrix_multiply(model.init_hiddens, model.weights['hidden_hidden'], model.bias['hidden'])
        if model.nonlinearity == "sigmoid":
            expit(hiddens[0,:], hiddens[0,:]) #sigmoid
        elif model.nonlinearity == "tanh":
            np.tanh(hiddens[0,:], hiddens[0,:])
        else: #relu
            hiddens[0,:] *= 0.5 * (np.sign(hiddens[0,:]) + 1)
        
        for time_step in range(1, num_observations):
            hiddens[time_step:time_step+1,:] += self.weight_matrix_multiply(hiddens[time_step-1:time_step,:], model.weights['hidden_hidden'], model.bias['hidden'])
            if model.nonlinearity == "sigmoid":
                expit(hiddens[time_step,:], hiddens[time_step,:]) #sigmoid
            elif model.nonlinearity == "tanh":
                np.tanh(hiddens[time_step,:], hiddens[time_step,:])
            else: #relu
                hiddens[time_step,:] *= 0.5 * (np.sign(hiddens[time_step,:]) + 1)
        
        if 'visible_output' in model.weights:
#            print inputs.dot(model.weights['visible_output']).shape
            outputs = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']) + 
                                   inputs.dot(model.weights['visible_output']))
        else:
            outputs = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']))
        
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
            return outputs
            
    def forward_pass(self, inputs, feature_sequence_lens, model=None, return_hiddens=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = max(feature_sequence_lens)
        num_sequences = len(feature_sequence_lens)
        num_hiddens = architecture[1]
        num_outs = architecture[2]
        if return_hiddens:
            hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        #propagate hiddens
        cur_index = 0
        end_index = cur_index
        for sequence_index, feature_sequence_len in enumerate(feature_sequence_lens):
            end_index += feature_sequence_len
            if not return_hiddens:
                outputs[:feature_sequence_len, sequence_index, :] = self.forward_pass_single_batch(inputs[cur_index:end_index], model, 
                                                                                                   return_hiddens)
            else:
                outputs[:feature_sequence_len, sequence_index, :], hiddens[:feature_sequence_len, sequence_index, :] = self.forward_pass_single_batch(inputs[cur_index:end_index], model, 
                                                                                                                                                      return_hiddens)
            cur_index = end_index

        if return_hiddens:
            return outputs, hiddens
        else:
#            del hiddens
            return outputs

    def flatten_output(self, output, feature_sequence_lens=None):
        """outputs in the form of max_obs_seq x n_seq x n_outs  get converted to form
        n_data x n_outs, so we can calculate classification accuracy and cross-entropy
        """
        if feature_sequence_lens == None:
            feature_sequence_lens = self.feature_sequence_lens
        num_outs = output.shape[2]
#        num_seq = output.shape[1]
        flat_output = np.zeros((self.batch_size(feature_sequence_lens), num_outs))
        cur_index = 0
        for seq_index, num_obs in enumerate(feature_sequence_lens):
            flat_output[cur_index:cur_index+num_obs, :] = copy.deepcopy(output[:num_obs, seq_index, :])
            cur_index += num_obs
        
        return flat_output

    def calculate_log_perplexity(self, output, flat_labels): #completed, expensive, should be compiled
        """calculates perplexity with flat labels
        """
        return -np.sum(np.log2(np.clip(output, a_min=1E-12, a_max=1.0))[np.arange(flat_labels.shape[0]), flat_labels[:,1]])

    def calculate_cross_entropy(self, output, flat_labels): #completed, expensive, should be compiled
        """calculates perplexity with flat labels
        """
        return -np.sum(np.log(np.clip(output, a_min=1E-12, a_max=1.0))[np.arange(flat_labels.shape[0]), flat_labels[:,1]])

    def calculate_classification_accuracy(self, flat_output, labels): #completed, possibly expensive
        prediction = flat_output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = sum(prediction == labels) / float(labels.size)
        return classification_accuracy[0]
    

class DSSMNNLM_Tester(DSSM_Neural_Network_Language_Model): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(DSSMNNLM_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.model.open_weights(self.weight_matrix_name)
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string',error_string="No label_file_name defined, just running forward pass",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
#            self.labels, self.labels_sent_id = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name
        self.dump_config_vals()
        self.classify()
        self.write_posterior_prob_file()
    
    def classify(self): #completed
        self.posterior_probs = self.forward_pass(self.features, self.feature_sequence_lens)
        
        if hasattr(self, 'labels'):
            self.flat_posterior_probs = self.flatten_output(self.posterior_probs, self.feature_sequence_lens)
            avg_cross_entropy = self.calculate_cross_entropy(self.flat_posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is %f%%" % self.calculate_classification_accuracy(self.flat_posterior_probs, self.labels) * 100
        else:
            print "no labels given, so skipping classification statistics"
            
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs, 'sequence_lengths' : self.feature_sequence_lens}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()

class DSSMNNLM_Trainer(DSSM_Neural_Network_Language_Model):
    def __init__(self,config_dictionary): #completed
        """variables in NN_trainer object are:
        mode (set to 'train')
        feature_file_name - inherited from Neural_Network class, name of feature file (in .mat format with variable 'features' in it) to read from
        features - inherited from Neural_Network class, features
        label_file_name - name of label file (in .mat format with variable 'labels' in it) to read from
        labels - labels for backprop
        architecture - specified by n_hid, n_hid, ..., n_hid. # of feature dimensions and # of classes need not be specified
        weight_matrix_name - initial weight matrix, if specified, if not, will initialize from random
        initial_weight_max - needed if initial weight matrix not loaded
        initial_weight_min - needed if initial weight matrix not loaded
        initial_bias_max - needed if initial weight matrix not loaded
        initial_bias_min - needed if initial weight matrix not loaded
        l2_regularization_constant - strength of l2 (weight decay) regularization
        steepest_learning_rate - learning rate for steepest_descent backprop
        output_name - name of weight file to store to.
        ********************************************************************************
         At bare minimum, you'll need these variables set to train
         feature_file_name
         output_name
         this will run logistic regression using steepest descent, which is a bad idea"""
        
        #Raise error if we encounter under/overflow during training, because this is bad... code should handle this gracefully
        old_settings = np.seterr(over='raise',under='raise',invalid='raise')
        
        self.mode = 'train'
        super(DSSMNNLM_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.batch_size(self.feature_sequence_lens)
        self.num_sequences = self.feature_sequence_lens.size
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
#            self.labels, self.labels_sent_id = self.read_label_file()
            self.check_labels()
#            self.unflattened_labels = self.unflatten_labels(self.labels, self.labels_sent_id) 
        else:
            del self.label_file_name        

        self.validation_feature_file_name = self.default_variable_define(config_dictionary, 'validation_feature_file_name', arg_type='string', exit_if_no_default = False)
        if self.validation_feature_file_name is not None:
            self.validation_features, self.validation_fsl = self.read_feature_file(self.validation_feature_file_name)

        self.validation_label_file_name = self.default_variable_define(config_dictionary, 'validation_label_file_name', arg_type='string', exit_if_no_default = False)
        if self.validation_label_file_name is not None:
            self.validation_labels = self.read_label_file(self.validation_label_file_name)
        #initialize weights
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', exit_if_no_default=False)
        
        self.use_maxent = self.default_variable_define(config_dictionary, 'use_maxent', arg_type='boolean', default_value=False)
        
        self.nonlinearity = self.default_variable_define(config_dictionary, 'nonlinearity', arg_type='string', default_value='sigmoid',
                                                         acceptable_values = ['sigmoid', 'relu', 'tanh'])
        if self.weight_matrix_name != None:
            print "Since weight_matrix_name is defined, ignoring possible value of hiddens_structure"
            self.model.open_weights(self.weight_matrix_name)
        else: #initialize model
            del self.weight_matrix_name
            
            self.num_hiddens = self.default_variable_define(config_dictionary, 'num_hiddens', arg_type='int', exit_if_no_default=True)
            architecture = [self.features.shape[1], self.num_hiddens] #+1 because index starts at 0
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels[:,1])+1) #will have to change later if I have soft weights
#                architecture.append(np.max(self.labels.ravel())+1) #will have to change later if I have soft weights
            
            self.seed = self.default_variable_define(config_dictionary, 'seed', 'int', '0')
#            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
#            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
#            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=0.1)
#            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-0.1)
#            self.use_maxent = self.default_variable_define(config_dictionary, 'use_maxent', arg_type='boolean', default_value=False)
#            self.nonlinearity = self.default_variable_define(config_dictionary, 'nonlinearity', arg_type='string', default_value='sigmoid',
#                                                             acceptable_values = ['sigmoid', 'relu', 'tanh'])
            self.model.init_random_weights(architecture, 
#                                           self.initial_bias_max, self.initial_bias_min, 
#                                           self.initial_weight_min, self.initial_weight_max, 
                                           maxent=self.use_maxent,
                                           nonlinearity = self.nonlinearity, seed = self.seed)
            del architecture #we have it in the model
        #
        
        self.save_each_epoch = self.default_variable_define(config_dictionary, 'save_each_epoch', arg_type='boolean', default_value=False)
                    
        #backprop configuration
        if not hasattr(self, 'labels'):
            print "No labels found... cannot do backprop... Exiting now"
            sys.exit()
        self.l2_regularization_const = self.default_variable_define(config_dictionary, 'l2_regularization_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
        self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.08, 0.04, 0.02, 0.01], arg_type='float_comma_string')
        self.momentum_rate = self.default_variable_define(config_dictionary, 'momentum_rate', arg_type='float_comma_string', exit_if_no_default=False)
        if hasattr(self, 'momentum_rate'):
            assert(len(self.momentum_rate) == len(self.steepest_learning_rate))
        self.dump_config_vals()
    
    def calculate_deriv_layer(self, layer, nonlinearity):
        if nonlinearity == "sigmoid": return layer * (1 - layer)
        elif nonlinearity == "tanh": return (1 + layer) * (1 - layer)
        elif nonlinearity == "relu": return (np.sign(layer) + 1) / 2
        else: raise ValueError('nonlinearity is sigmoid, tanh, or relu')
    
    def update_by_csr_matrix(self, csr_mat, dense_mat, out_mat):
        for obs_index in range(csr_mat.shape[0]):
            start_index = csr_mat.indptr[obs_index]
            end_index = csr_mat.indptr[obs_index+1]
            out_mat[csr_mat.indices[start_index:end_index]] += dense_mat[obs_index]
            
    def calculate_gradient_single_batch(self, batch_inputs, batch_labels, gradient_weights, hiddens = None, outputs = None, check_gradient=False, 
                                        model=None, return_cross_entropy = False): 
        #need to check regularization
        batch_size = batch_labels.size
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass_single_batch(batch_inputs, model, return_hiddens=True)
        #derivative of log(cross-entropy softmax)
        batch_indices = np.arange(batch_size)
        gradient_weights *= 0.0
        backward_inputs = outputs
        if return_cross_entropy:
            cross_entropy = -np.sum(np.log2(backward_inputs[batch_indices, batch_labels]))
        backward_inputs[batch_indices, batch_labels] -= 1.0
        pre_nonlinearity_hiddens = np.empty((batch_size, model.weights['visible_hidden'].shape[1]))
        hiddens_nonlinearity = self.calculate_deriv_layer(hiddens, model.nonlinearity)

        np.sum(backward_inputs, axis=0, out = gradient_weights.bias['output'][0])
        np.dot(hiddens.T, backward_inputs, out = gradient_weights.weights['hidden_output'])
        if 'visible_output' in model.weights:
            gradient_weights.weights['visible_output'][:] += batch_inputs.T.dot(backward_inputs)
        pre_nonlinearity_hiddens[batch_size-1,:] = np.dot(backward_inputs[batch_size-1,:], model.weights['hidden_output'].T)
        
        pre_nonlinearity_hiddens[batch_size-1,:] *= hiddens_nonlinearity[batch_size-1,:] 

        if batch_size > 1:
            gradient_weights.weights['hidden_hidden'] += np.outer(hiddens[batch_size-2,:], pre_nonlinearity_hiddens[batch_size-1,:])
            
        for observation_index in range(batch_size-2,0,-1):
            pre_nonlinearity_hiddens[observation_index,:] = ((np.dot(backward_inputs[observation_index,:], model.weights['hidden_output'].T) + 
                                                              np.dot(pre_nonlinearity_hiddens[observation_index+1,:], model.weights['hidden_hidden'].T))
                                                             * hiddens_nonlinearity[observation_index,:])

            gradient_weights.weights['hidden_hidden'] += np.outer(hiddens[observation_index-1,:], pre_nonlinearity_hiddens[observation_index,:])

        
        if batch_size > 1:
            pre_nonlinearity_hiddens[0,:] = ((np.dot(backward_inputs[0,:], model.weights['hidden_output'].T) 
                                              + np.dot(pre_nonlinearity_hiddens[1,:], model.weights['hidden_hidden'].T))
                                             * hiddens_nonlinearity[0,:])

        gradient_weights.weights['hidden_hidden'] += np.outer(model.init_hiddens, pre_nonlinearity_hiddens[0,:]) #np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        gradient_weights.init_hiddens[0] = np.dot(pre_nonlinearity_hiddens[0,:], model.weights['hidden_hidden'].T)
        gradient_weights.bias['hidden'][0] = np.sum(pre_nonlinearity_hiddens,0)
#        print batch_inputs.data
#        print batch_inputs.indices
#        print batch_inputs.indptr
#        print batch_inputs.shape
#        print np.where(batch_inputs.todense() > 0)
#        print pre_nonlinearity_hiddens.shape

        self.update_by_csr_matrix(batch_inputs, pre_nonlinearity_hiddens, gradient_weights.weights['visible_hidden'])
#        gradient_weights.weights['visible_hidden'][:] = batch_inputs.T.dot(pre_nonlinearity_hiddens)
        backward_inputs[batch_indices, batch_labels] += 1.0
        gradient_weights /= batch_size
        
        if return_cross_entropy and not check_gradient:
            return cross_entropy
        elif not check_gradient:
            return
            
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            self.check_gradient(gradient_weights, batch_inputs, batch_labels, model)
            
    def check_gradient(self, gradient_weights, batch_inputs, batch_labels, model):
        sys.stdout.write("\r                                                                \r")
        batch_size = batch_labels.size
        batch_indices = np.arange(batch_size)
        print "checking gradient..."
        finite_difference_model = RNNLM_Weight()
        finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False, nonlinearity = self.nonlinearity)
        
        direction = RNNLM_Weight()
        direction.init_zero_weights(self.model.get_architecture(), verbose=False, nonlinearity = self.nonlinearity)
        epsilon = 1E-5
        print "at initial hiddens"
        for index in range(direction.init_hiddens.size):
            direction.init_hiddens[0][index] = epsilon
            forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
            backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
            finite_difference_model.init_hiddens[0][index] = (forward_loss - backward_loss) / (2 * epsilon)
            direction.init_hiddens[0][index] = 0.0
        for key in direction.bias.keys():
            print "at bias key", key
            for index in range(direction.bias[key].size):
                direction.bias[key][0][index] = epsilon
                #print direction.norm()
                forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
                backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
                finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                direction.bias[key][0][index] = 0.0
        for key in direction.weights.keys():
            print "at weight key", key
            for index0 in range(direction.weights[key].shape[0]):
                for index1 in range(direction.weights[key].shape[1]):
                    direction.weights[key][index0][index1] = epsilon
                    forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
                    backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
                    finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.weights[key][index0][index1] = 0.0
        
        print "calculated gradient for initial hiddens"
        print gradient_weights.init_hiddens
        print "finite difference approximation for initial hiddens"
        print finite_difference_model.init_hiddens
        
        print "calculated gradient for hidden bias"
        print gradient_weights.bias['hidden']
        print "finite difference approximation for hidden bias"
        print finite_difference_model.bias['hidden']
        
        print "calculated gradient for output bias"
        print gradient_weights.bias['output']
        print "finite difference approximation for output bias"
        print finite_difference_model.bias['output']
        
        print "calculated gradient for visible_hidden layer"
        print gradient_weights.weights['visible_hidden']
        print "finite difference approximation for visible_hidden layer"
        print finite_difference_model.weights['visible_hidden']
        
        print "calculated gradient for hidden_hidden layer"
        print gradient_weights.weights['hidden_hidden']
        print "finite difference approximation for hidden_hidden layer"
        print finite_difference_model.weights['hidden_hidden']
        
        print "calculated gradient for hidden_output layer"
        print gradient_weights.weights['hidden_output']
        print "finite difference approximation for hidden_output layer"
        print finite_difference_model.weights['hidden_output']
        
        sys.exit()
        ##########################################################
        
    def calculate_classification_statistics(self, features, flat_labels, feature_sequence_lens, model=None, classification_batch_size = 64):
        if model == None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': []}
        
        batch_index = 0
        end_index = 0
        cross_entropy = 0.0
        log_perplexity = 0.0
        num_correct = 0
        num_sequences = len(feature_sequence_lens)
        num_examples = self.batch_size(feature_sequence_lens)
#        print features.shape
        print "calculating classification statistics"
        while end_index < num_sequences: #run through the batches
#            per_done = float(batch_index)/num_sequences*100
#            sys.stdout.write("\r                                                                \r") #clear line
#            sys.stdout.write("\rCalculating Classification Statistics: %.1f%% done " % per_done), sys.stdout.flush()
            end_index = min(batch_index+classification_batch_size, num_sequences)
            start_frame = np.where(flat_labels[:,0] == batch_index)[0][0]
            end_frame = np.where(flat_labels[:,0] == end_index-1)[0][-1] + 1
            label = flat_labels[start_frame:end_frame]
#            print batch_index, max_seq_len
            output = self.flatten_output(self.forward_pass(features[start_frame:end_frame], feature_sequence_lens[batch_index:end_index], model=model), 
                                         feature_sequence_lens[batch_index:end_index])
            
            cross_entropy += self.calculate_cross_entropy(output, label)
            log_perplexity += self.calculate_log_perplexity(output, label)
            #don't use calculate_classification_accuracy() because of possible rounding error
            prediction = output.argmax(axis=1)
            num_correct += np.sum(prediction == label[:,1]) #- (prediction.size - num_examples) #because of the way we handle features, where some observations are null, we want to remove those examples for calculating accuracy
            batch_index += classification_batch_size
        
#        sys.stdout.write("\r                                                                \r") #clear line
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        
        loss /= np.log(2) * num_examples
        log_perplexity /= num_examples
        perplexity = 2 ** log_perplexity
        return cross_entropy, perplexity, num_correct, num_examples, loss

    def backprop_steepest_descent_single_batch(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent, nonlinearity = self.nonlinearity)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent, nonlinearity = self.nonlinearity)
        
        if self.validation_feature_file_name is not None:
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            print "perplexity is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)

        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            print "Training for epoch started at", datetime.datetime.now()
            start_frame = 0
            end_frame = 0
            cross_entropy = 0.0
            num_examples = 0
            if hasattr(self, 'momentum_rate'):
                momentum_rate = self.momentum_rate[epoch_num]
                print "momentum is", momentum_rate
            else:
                momentum_rate = 0.0
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                batch_features = self.features[start_frame:end_frame]
                batch_labels = self.labels[start_frame:end_frame,1]
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_labels, gradient, return_cross_entropy = True, 
                                                                check_gradient = False)
                cross_entropy += cur_xent 
#                per_done = float(batch_index)/self.num_sequences*100
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
#                ppp = cross_entropy / end_frame
#                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                gradient *= -self.steepest_learning_rate[epoch_num]
                if self.l2_regularization_const > 0.0:
                    self.model *= (1-self.l2_regularization_const) #l2 regularization_const
                self.model += gradient #/ batch_size
                if momentum_rate > 0.0:
                    prev_step *= momentum_rate
                    self.model += prev_step
                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num]
                
                start_frame = end_frame
            print "Training for epoch finished at", datetime.datetime.now()
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
                
#            sys.stdout.write("\r100.0% done \r")
#            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))           
            print "Epoch finished at", datetime.datetime.now()
        end_time = datetime.datetime.now()
        print "Training finished at", end_time, "and ran for", end_time - start_time
    
    def backprop_steepest_descent_single_batch_semi_newbob(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent, nonlinearity = self.nonlinearity)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent, nonlinearity = self.nonlinearity)
        
        if self.validation_feature_file_name is not None:
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            print "perplexity is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
        learning_rate = self.steepest_learning_rate[0]
        if hasattr(self, 'momentum_rate'):
            momentum_rate = self.momentum_rate[0]
            print "momentum is", momentum_rate
        else:
            momentum_rate = 0.0
        num_decreases = 0
        prev_cross_entropy = cross_entropy
        prev_num_correct = num_correct
        for epoch_num in range(100):
            print "At epoch", epoch_num+1, "with learning rate", learning_rate, "and momentum", momentum_rate
            print "Training for epoch started at", datetime.datetime.now()
            start_frame = 0
            end_frame = 0
            cross_entropy = 0.0
            num_examples = 0
            
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                batch_features = self.features[start_frame:end_frame]
                batch_labels = self.labels[start_frame:end_frame,1]
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_labels, gradient, return_cross_entropy = True, 
                                                                check_gradient = False)
                cross_entropy += cur_xent 
#                per_done = float(batch_index)/self.num_sequences*100
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
#                ppp = cross_entropy / end_frame
#                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                gradient *= -learning_rate
                if self.l2_regularization_const > 0.0:
                    self.model *= (1-self.l2_regularization_const) #l2 regularization_const
                self.model += gradient #/ batch_size
                if momentum_rate > 0.0:
                    prev_step *= momentum_rate
                    self.model += prev_step
                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num]
                
                start_frame = end_frame
            print "Training for epoch finished at", datetime.datetime.now()
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
            else:
                raise ValueError("validation feature file must exist")
            if prev_num_correct < num_correct:
                prev_cross_entropy = cross_entropy
                prev_num_correct = num_correct
                self.model.write_weights(''.join([self.output_name, '_best_weights']))
                if self.save_each_epoch:
                    self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
                print "num_decreases so far is", num_decreases
                if num_decreases == 2: 
                    learning_rate /= 2.0
                    momentum_rate /= 2.0
            else:
                num_decreases += 1
                print "cross-entropy did not decrease, so using previous best weights"
                self.model.open_weights(''.join([self.output_name, '_best_weights']))
                if num_decreases > 2: break
                learning_rate /= 2.0
                momentum_rate /= 2.0
#            sys.stdout.write("\r100.0% done \r")
#            sys.stdout.write("\r                                                                \r") #clear line           
            print "Epoch finished at", datetime.datetime.now()
        self.model.write_weights(self.output_name)
        end_time = datetime.datetime.now()
        print "Training finished at", end_time, "and ran for", end_time - start_time