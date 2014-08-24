'''
Created on Jun 3, 2013

@author: sumanravuri
'''

import sys
import numpy as np
import scipy.io as sp
import copy
import argparse
from Vector_Math import Vector_Math
import datetime
from scipy.special import expit
from RNNLM_Weight import RNNLM_Weight

class Recurrent_Neural_Network_Language_Model(object, Vector_Math):
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
                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 'save_each_epoch',
                               'do_pretrain', 'pretrain_method', 'pretrain_iterations', 
                               'pretrain_learning_rate', 'pretrain_batch_size',
                               'do_backprop', 'backprop_method', 'backprop_batch_size', 'l2_regularization_const',
                               'num_epochs', 'num_line_searches', 'armijo_const', 'wolfe_const',
                               'steepest_learning_rate', 'momentum_rate',
                               'conjugate_max_iterations', 'conjugate_const_type',
                               'truncated_newton_num_cg_epochs', 'truncated_newton_init_damping_factor',
                               'krylov_num_directions', 'krylov_num_batch_splits', 'krylov_num_bfgs_epochs', 'second_order_matrix',
                               'krylov_use_hessian_preconditioner', 'krylov_eigenvalue_floor_const', 
                               'fisher_preconditioner_floor_val', 'use_fisher_preconditioner',
                               'structural_damping_const', 
                               'validation_feature_file_name', 'validation_label_file_name']
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
            features = feature_data['features'].astype(np.int32)
            sequence_len = feature_data['feature_sequence_lengths']
            sequence_len = np.reshape(sequence_len, (sequence_len.size,))
            return features, sequence_len#in MATLAB format
        except IOError:
            print "Unable to open ", feature_file_name, "... Exiting now"
            sys.exit()
    
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
            print "Number of examples in feature file: ", sum(self.feature_sequence_lens), " does not equal size of label file, ", self.labels.size, "... Exiting now"
            sys.exit()
#        if  [i for i in np.unique(self.labels)] != range(np.max(self.labels)+1):
#            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
            sys.exit()
#        label_counts = np.bincount(np.ravel(self.labels[:,1])) #[self.labels.count(x) for x in range(np.max(self.labels)+1)]
#        print "distribution of labels is:"
#        for x in range(len(label_counts)):
#            print "#", x, "\b's:", label_counts[x]            
        print "labels seem copacetic"

    def forward_layer(self, inputs, weights, biases, weight_type, prev_hiddens = None, hidden_hidden_weights = None): #completed
        if weight_type == 'logistic':
            return self.softmax(self.weight_matrix_multiply(inputs, weights, biases))
        elif weight_type == 'rbm_gaussian_bernoulli' or weight_type == 'rbm_bernoulli_bernoulli':
            return self.sigmoid(weights[(inputs),:] + self.weight_matrix_multiply(prev_hiddens, hidden_hidden_weights, biases))
        #added to test finite differences calculation for pearlmutter forward pass
        elif weight_type == 'linear': #only used for the logistic layer
            return self.weight_matrix_multiply(inputs, weights, biases)
        else:
            print "weight_type", weight_type, "is not a valid layer type.",
            print "Valid layer types are", self.model.valid_layer_types,"Exiting now..."
            sys.exit()
            
#    def forward_pass_linear(self, inputs, verbose=True, model=None):
#        #to test finite differences calculation for pearlmutter forward pass, just like forward pass, except it spits linear outputs
#        if model == None:
#            model = self.model
#        architecture = self.model.get_architecture()
#        max_sequence_observations = inputs.shape[0]
#        num_hiddens = architecture[1]
#        num_sequences = inputs.shape[2]
#        num_outs = architecture[2]
#        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
#        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
#        
#        #propagate hiddens
#        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                            self.model.weight_type['visible_hidden'], self.model.init_hiddens, 
#                                            self.model.weights['hidden_hidden'])
#        outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                            self.model.weight_type['hidden_output'])
#        for sequence_index in range(1, max_sequence_observations):
#            sequence_input = inputs[sequence_index,:,:]
#            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                                             self.model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
#                                                             self.model.weights['hidden_hidden'])
#            outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                                             'linear')
#            #find the observations where the sequence has ended, 
#            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
#            zero_input = np.where(self.feature_sequence_lens >= sequence_index)
#            hiddens[sequence_index,:,zero_input] = 0.0
#            outputs[sequence_index,:,zero_input] = 0.0
#
#        del hiddens
#        return outputs
    def forward_pass_single_batch(self, inputs, model = None, return_hiddens = False, linear_output = False):
        """forward pass for single batch size. Mainly for speed in this case
        """
        if model == None:
            model = self.model
        num_observations = inputs.size
        hiddens = model.weights['visible_hidden'][(inputs),:]
        hiddens[:1,:] += self.weight_matrix_multiply(model.init_hiddens, model.weights['hidden_hidden'], model.bias['hidden'])
        expit(hiddens[0,:], hiddens[0,:])
        
        for time_step in range(1, num_observations):
            hiddens[time_step:time_step+1,:] += self.weight_matrix_multiply(hiddens[time_step-1:time_step,:], model.weights['hidden_hidden'], model.bias['hidden'])
            expit(hiddens[time_step,:], hiddens[time_step,:]) #sigmoid
        
        outputs = self.forward_layer(hiddens, model.weights['hidden_output'], model.bias['output'], model.weight_type['hidden_output'])
        
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
            return outputs
            
    def forward_pass(self, inputs, feature_sequence_lens, model=None, return_hiddens=False, linear_output=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_sequences = inputs.shape[1]
        num_hiddens = architecture[1]
        num_outs = architecture[2]
        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        #propagate hiddens
        hiddens[0,:,:] = self.forward_layer(inputs[0,:], model.weights['visible_hidden'], model.bias['hidden'], 
                                            model.weight_type['visible_hidden'], model.init_hiddens, 
                                            model.weights['hidden_hidden'])
        if linear_output:
            outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                'linear')
        else:
            outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                model.weight_type['hidden_output'])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:]
            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, model.weights['visible_hidden'], model.bias['hidden'], 
                                                             model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
                                                             model.weights['hidden_hidden'])
            if linear_output:
                outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                             'linear')
            else:
                outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                                 model.weight_type['hidden_output'])
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(feature_sequence_lens <= sequence_index)
            hiddens[sequence_index,zero_input,:] = 0.0
            outputs[sequence_index,zero_input,:] = 0.0
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
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
    

class RNNLM_Tester(Recurrent_Neural_Network_Language_Model): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(RNNLM_Tester,self).__init__(config_dictionary)
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
        self.flat_posterior_probs = self.flatten_output(self.posterior_probs)
        try:
            avg_cross_entropy = self.calculate_cross_entropy(self.flat_posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is %f%%" % self.calculate_classification_accuracy(self.flat_posterior_probs, self.labels) * 100
        except AttributeError:
            print "no labels given, so skipping classification statistics"
                
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs, 'sequence_lengths' : self.feature_sequence_lens}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()

class RNNLM_Trainer(Recurrent_Neural_Network_Language_Model):
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
        do_pretrain - set to 1 or 0 (probably should change to boolean values)
        pretrain_method - not yet implemented, will either be 'mean_field' or 'sampling'
        pretrain_iterations - # of iterations per RBM. Must be equal to the number of hidden layers
        pretrain_learning_rate - learning rate for each epoch of pretrain. must be equal to # hidden layers * sum(pretrain_iterations)
        pretrain_batch_size - batch size for pretraining
        do_backprop - do backpropagation (set to either 0 or 1, probably should be changed to boolean value)
        backprop_method - either 'steepest_descent', 'conjugate_gradient', or '2nd_order', latter two not yet implemented
        l2_regularization_constant - strength of l2 (weight decay) regularization
        steepest_learning_rate - learning rate for steepest_descent backprop
        backprop_batch_size - batch size for backprop
        output_name - name of weight file to store to.
        ********************************************************************************
         At bare minimum, you'll need these variables set to train
         feature_file_name
         output_name
         this will run logistic regression using steepest descent, which is a bad idea"""
        
        #Raise error if we encounter under/overflow during training, because this is bad... code should handle this gracefully
        old_settings = np.seterr(over='raise',under='raise',invalid='raise')
        
        self.mode = 'train'
        super(RNNLM_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.batch_size(self.feature_sequence_lens)
        self.num_sequences = self.features.shape[1]
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

        if self.weight_matrix_name != None:
            print "Since weight_matrix_name is defined, ignoring possible value of hiddens_structure"
            self.model.open_weights(self.weight_matrix_name)
        else: #initialize model
            del self.weight_matrix_name
            
            self.num_hiddens = self.default_variable_define(config_dictionary, 'num_hiddens', arg_type='int', exit_if_no_default=True)
            architecture = [np.max(self.features)+1, self.num_hiddens] #+1 because index starts at 0
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels[:,1])+1) #will have to change later if I have soft weights
                
            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=0.1)
            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-0.1)
            self.model.init_random_weights(architecture, self.initial_bias_max, self.initial_bias_min, 
                                           self.initial_weight_min, self.initial_weight_max)
            del architecture #we have it in the model
        #
        
        self.save_each_epoch = self.default_variable_define(config_dictionary, 'save_each_epoch', arg_type='boolean', default_value=False)
        #pretraining configuration
        self.do_pretrain = self.default_variable_define(config_dictionary, 'do_pretrain', default_value=False, arg_type='boolean')
        if self.do_pretrain:
            self.pretrain_method = self.default_variable_define(config_dictionary, 'pretrain_method', default_value='mean_field', acceptable_values=['mean_field', 'sampling'])
            self.pretrain_iterations = self.default_variable_define(config_dictionary, 'pretrain_iterations', default_value=[5] * len(self.hiddens_structure), 
                                                                    error_string="No pretrain_iterations defined, setting pretrain_iterations to default 5 per layer", 
                                                                    arg_type='int_comma_string')

            weight_last_layer = ''.join([str(self.model.num_layers-1), str(self.model.num_layers)])
            if self.model.weight_type[weight_last_layer] == 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers - 1):
                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers-1, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
                sys.exit()
            elif self.model.weight_type[weight_last_layer] != 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers):
                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
                sys.exit()
            self.pretrain_learning_rate = self.default_variable_define(config_dictionary, 'pretrain_learning_rate', default_value=[0.01] * sum(self.pretrain_iterations), 
                                                                       error_string="No pretrain_learning_rate defined, setting pretrain_learning_rate to default 0.01 per iteration", 
                                                                       arg_type='float_comma_string')
            if len(self.pretrain_learning_rate) != sum(self.pretrain_iterations):
                print "pretraining learning rate should have ", sum(self.pretrain_iterations), " learning rate iterations but only has ", len(self.pretrain_learning_rate), "... Exiting now"
                sys.exit()
            self.pretrain_batch_size = self.default_variable_define(config_dictionary, 'pretrain_batch_size', default_value=256, arg_type='int')
                    
        #backprop configuration
        self.do_backprop = self.default_variable_define(config_dictionary, 'do_backprop', default_value=True, arg_type='boolean')
        if self.do_backprop:
            if not hasattr(self, 'labels'):
                print "No labels found... cannot do backprop... Exiting now"
                sys.exit()
            self.backprop_method = self.default_variable_define(config_dictionary, 'backprop_method', default_value='steepest_descent', 
                                                                acceptable_values=['steepest_descent', 'adagrad', 'krylov_subspace', 'truncated_newton'])
            self.backprop_batch_size = self.default_variable_define(config_dictionary, 'backprop_batch_size', default_value=16, arg_type='int')
            self.l2_regularization_const = self.default_variable_define(config_dictionary, 'l2_regularization_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
            
            if self.backprop_method == 'steepest_descent':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.08, 0.04, 0.02, 0.01], arg_type='float_comma_string')
                self.momentum_rate = self.default_variable_define(config_dictionary, 'momentum_rate', arg_type='float_comma_string', exit_if_no_default=False)
                if hasattr(self, 'momentum_rate'):
                    assert(len(self.momentum_rate) == len(self.steepest_learning_rate))
            if self.backprop_method == 'adagrad':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.08, 0.04, 0.02, 0.01], arg_type='float_comma_string')
            else: #second order methods
                self.num_epochs = self.default_variable_define(config_dictionary, 'num_epochs', default_value=20, arg_type='int')
                self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
                self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
                                                                        acceptable_values=['gauss-newton', 'hessian', 'fisher'])
                self.structural_damping_const = self.default_variable_define(config_dictionary, 'structural_damping_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
                if self.use_fisher_preconditioner:
                    self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
                if self.backprop_method == 'krylov_subspace':
                    self.krylov_num_directions = self.default_variable_define(config_dictionary, 'krylov_num_directions', arg_type='int', default_value=20, 
                                                                              acceptable_values=range(2,2000))
                    self.krylov_num_batch_splits = self.default_variable_define(config_dictionary, 'krylov_num_batch_splits', arg_type='int', default_value=self.krylov_num_directions, 
                                                                                acceptable_values=range(2,2000))
                    self.krylov_num_bfgs_epochs = self.default_variable_define(config_dictionary, 'krylov_num_bfgs_epochs', arg_type='int', default_value=self.krylov_num_directions)
                    self.krylov_use_hessian_preconditioner = self.default_variable_define(config_dictionary, 'krylov_use_hessian_preconditioner', arg_type='boolean', default_value=True)
                    if self.krylov_use_hessian_preconditioner:
                        self.krylov_eigenvalue_floor_const = self.default_variable_define(config_dictionary, 'krylov_eigenvalue_floor_const', arg_type='float', default_value=1E-4)
                    
                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.0001)
                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.9)
                elif self.backprop_method == 'truncated_newton':
                    self.truncated_newton_num_cg_epochs = self.default_variable_define(config_dictionary, 'truncated_newton_num_cg_epochs', arg_type='int', default_value=20)
                    self.truncated_newton_init_damping_factor = self.default_variable_define(config_dictionary, 'truncated_newton_init_damping_factor', arg_type='float', default_value=0.1)
        self.dump_config_vals()
    
    def calculate_gradient_single_batch(self, batch_inputs, batch_labels, gradient_weights, hiddens = None, outputs = None, check_gradient=False, 
                                        model=None, l2_regularization_const = 0.0, return_cross_entropy = False): 
        #need to check regularization
        #TO DO: fix gradient when there is only a single word (empty?)
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        batch_size = batch_labels.size
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass_single_batch(batch_inputs, model, return_hiddens=True)
        #derivative of log(cross-entropy softmax)
        batch_indices = np.arange(batch_size)
        gradient_weights *= 0.0
        backward_inputs = outputs
#        print batch_inputs
#        print batch_labels
#        print batch_indices
        if return_cross_entropy:
            cross_entropy = -np.sum(np.log2(backward_inputs[batch_indices, batch_labels]))
        backward_inputs[batch_indices, batch_labels] -= 1.0
#        print backward_inputs.shape
#        gradient_weights = RNNLM_Weight()
#        gradient_weights.init_zero_weights(self.model.get_architecture(), verbose=False)
        
#        gradient_weights.bias['output'][0] = np.sum(backward_inputs, axis=0)
        np.sum(backward_inputs, axis=0, out = gradient_weights.bias['output'][0])
        np.dot(hiddens.T, backward_inputs, out = gradient_weights.weights['hidden_output'])
#        backward_inputs = outputs - batch_unflattened_labels
        pre_nonlinearity_hiddens = np.dot(backward_inputs[batch_size-1,:], model.weights['hidden_output'].T) 
        pre_nonlinearity_hiddens *= hiddens[batch_size-1,:] 
        pre_nonlinearity_hiddens *= 1 - hiddens[batch_size-1,:]
#        if structural_damping_const > 0.0:
#            pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[n_obs-1,:,:]
#        output_model.weights['visible_hidden'] += np.dot(visibles[n_obs-1,:,:].T, pre_nonlinearity_hiddens)
        if batch_size > 1:
            gradient_weights.weights['visible_hidden'][batch_inputs[batch_size-1]] += pre_nonlinearity_hiddens
            gradient_weights.weights['hidden_hidden'] += np.outer(hiddens[batch_size-2,:], pre_nonlinearity_hiddens)
            gradient_weights.bias['hidden'][0] += pre_nonlinearity_hiddens
        for observation_index in range(batch_size-2,0,-1):
            pre_nonlinearity_hiddens = ((np.dot(backward_inputs[observation_index,:], model.weights['hidden_output'].T) + 
                                         np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                        * hiddens[observation_index,:] * (1 - hiddens[observation_index,:]))
#            np.dot(backward_inputs[observation_index,:], model.weights['hidden_output'].T, out = pre_nonlinearity_hiddens)
#            pre_nonlinearity_hiddens += np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T)
#            pre_nonlinearity_hiddens *= hiddens[observation_index,:] 
#            pre_nonlinearity_hiddens *= (1 - hiddens[observation_index,:])
#            print pre_nonlinearity_hiddens.shape
#            if structural_damping_const > 0.0:
#                pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[observation_index,:,:]
            gradient_weights.weights['visible_hidden'][batch_inputs[observation_index]] += pre_nonlinearity_hiddens #+= np.dot(visibles[observation_index,:,:].T, pre_nonlinearity_hiddens)
            gradient_weights.weights['hidden_hidden'] += np.outer(hiddens[observation_index-1,:], pre_nonlinearity_hiddens)
            gradient_weights.bias['hidden'][0] += pre_nonlinearity_hiddens
        
        if batch_size > 1:
            pre_nonlinearity_hiddens = ((np.dot(backward_inputs[0,:], model.weights['hidden_output'].T) 
                                         + np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                        * hiddens[0,:] * (1 - hiddens[0,:]))
        gradient_weights.weights['visible_hidden'][batch_inputs[0]] += pre_nonlinearity_hiddens# += np.dot(visibles[0,:,:].T, pre_nonlinearity_hiddens)
        gradient_weights.weights['hidden_hidden'] += np.outer(model.init_hiddens, pre_nonlinearity_hiddens) #np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        gradient_weights.bias['hidden'][0] += pre_nonlinearity_hiddens
        gradient_weights.init_hiddens[0] = np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T)
#        gradient_weights = self.backward_pass(backward_inputs, hiddens, batch_inputs, model)
        backward_inputs[batch_indices, batch_labels] += 1.0
        gradient_weights /= batch_size
        
        if l2_regularization_const > 0.0:
            gradient_weights += model * l2_regularization_const
        if return_cross_entropy and not check_gradient:
            return cross_entropy
#        if not check_gradient:
#            if not return_cross_entropy:
#                if l2_regularization_const > 0.0:
#                    return gradient_weights / batch_size + model * l2_regularization_const
#                return gradient_weights / batch_size
#            else:
#                if l2_regularization_const > 0.0:
#                    return gradient_weights / batch_size + model * l2_regularization_const, cross_entropy
#                return gradient_weights / batch_size, cross_entropy
            
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * (l2_regularization_const * batch_size)
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = RNNLM_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
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
            print np.sum((finite_difference_model.weights['visible_hidden'] - gradient_weights.weights['visible_hidden']) ** 2)
            
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
                                   
    def calculate_gradient(self, batch_inputs, batch_labels, feature_sequence_lens, hiddens = None, outputs = None,
                           check_gradient=False, model=None, l2_regularization_const = 0.0, return_cross_entropy = False): 
        #need to check regularization
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(batch_inputs, feature_sequence_lens, model, return_hiddens=True)
        #derivative of log(cross-entropy softmax)
        batch_size = self.batch_size(feature_sequence_lens)
        batch_indices = np.zeros((batch_size,), dtype=np.int)
        cur_frame = 0
        for seq_len in feature_sequence_lens:
            batch_indices[cur_frame:cur_frame+seq_len] = np.arange(seq_len)
            cur_frame += seq_len
            
        backward_inputs = copy.deepcopy(outputs)
#        print batch_labels
#        print batch_indices
        if return_cross_entropy:
            cross_entropy = -np.sum(np.log2(backward_inputs[batch_indices, batch_labels[:,0], batch_labels[:,1]]))
        backward_inputs[batch_indices, batch_labels[:,0], batch_labels[:,1]] -= 1.0
        
#        backward_inputs = outputs - batch_unflattened_labels

        gradient_weights = self.backward_pass(backward_inputs, hiddens, batch_inputs, model)
        
        if not check_gradient:
            if not return_cross_entropy:
                if l2_regularization_const > 0.0:
                    return gradient_weights / batch_size + model * l2_regularization_const
                return gradient_weights / batch_size
            else:
                if l2_regularization_const > 0.0:
                    return gradient_weights / batch_size + model * l2_regularization_const, cross_entropy
                return gradient_weights / batch_size, cross_entropy
        
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * (l2_regularization_const * batch_size)
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = RNNLM_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
            print "at initial hiddens"
            for index in range(direction.init_hiddens.size):
                direction.init_hiddens[0][index] = epsilon
                forward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model + direction), 
                                                                                feature_sequence_lens), batch_labels)
                backward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model - direction), 
                                                                                feature_sequence_lens), batch_labels)
                finite_difference_model.init_hiddens[0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                direction.init_hiddens[0][index] = 0.0
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
                    forward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model + direction), 
                                                                                    feature_sequence_lens), batch_labels)
                    backward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model - direction), 
                                                                                     feature_sequence_lens), batch_labels)
                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        forward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model + direction), 
                                                                                        feature_sequence_lens), batch_labels)
                        backward_loss = self.calculate_cross_entropy(self.flatten_output(self.forward_pass(batch_inputs, feature_sequence_lens, model = model - direction), 
                                                                                         feature_sequence_lens), batch_labels)
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

    def backward_pass(self, backward_inputs, hiddens, visibles, model=None, structural_damping_const = 0.0, hidden_deriv = None): #need to test
        
        if model == None:
            model = self.model
        output_model = RNNLM_Weight()
        output_model.init_zero_weights(self.model.get_architecture(), verbose=False)
        
        n_obs = backward_inputs.shape[0]
        n_outs = backward_inputs.shape[2]
        n_hids = hiddens.shape[2]
        n_seq = backward_inputs.shape[1]
        #backward_inputs - n_obs x n_seq x n_outs
        #hiddens - n_obs  x n_seq x n_hids
        flat_outputs = np.reshape(np.transpose(backward_inputs, axes=(1,0,2)),(n_obs * n_seq,n_outs))
        flat_hids = np.reshape(np.transpose(hiddens, axes=(1,0,2)),(n_obs * n_seq,n_hids))
        #average layers in batch
        
        output_model.bias['output'][0] = np.sum(flat_outputs, axis=0)
        output_model.weights['hidden_output'] = np.dot(flat_hids.T, flat_outputs)
        
        #HERE is where the fun begins... will need to store gradient updates in the form of dL/da_i 
        #(where a is the pre-nonlinear layer for updates to the hidden hidden and input hidden weight matrices
        
        pre_nonlinearity_hiddens = np.dot(backward_inputs[n_obs-1,:,:], model.weights['hidden_output'].T) * hiddens[n_obs-1,:,:] * (1 - hiddens[n_obs-1,:,:])
        if structural_damping_const > 0.0:
            pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[n_obs-1,:,:]
#        output_model.weights['visible_hidden'] += np.dot(visibles[n_obs-1,:,:].T, pre_nonlinearity_hiddens)
        output_model.weights['visible_hidden'][visibles[n_obs-1,:]] += pre_nonlinearity_hiddens
        output_model.weights['hidden_hidden'] += np.dot(hiddens[n_obs-2,:,:].T, pre_nonlinearity_hiddens)
        output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        for observation_index in range(1,n_obs-1)[::-1]:
            pre_nonlinearity_hiddens = ((np.dot(backward_inputs[observation_index,:,:], model.weights['hidden_output'].T) + np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                        * hiddens[observation_index,:,:] * (1 - hiddens[observation_index,:,:]))
            if structural_damping_const > 0.0:
                pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[observation_index,:,:]
            output_model.weights['visible_hidden'][visibles[observation_index,:]] += pre_nonlinearity_hiddens #+= np.dot(visibles[observation_index,:,:].T, pre_nonlinearity_hiddens)
            output_model.weights['hidden_hidden'] += np.dot(hiddens[observation_index-1,:,:].T, pre_nonlinearity_hiddens)
            output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        
        pre_nonlinearity_hiddens = ((np.dot(backward_inputs[0,:,:], model.weights['hidden_output'].T) + np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                    * hiddens[0,:,:] * (1 - hiddens[0,:,:]))
        output_model.weights['visible_hidden'][visibles[0,:]] += pre_nonlinearity_hiddens# += np.dot(visibles[0,:,:].T, pre_nonlinearity_hiddens)
        output_model.weights['hidden_hidden'] += np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        output_model.init_hiddens[0] = np.sum(np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T), axis=0)
        
        return output_model
    
    def calculate_loss(self, inputs, feature_sequence_lens, labels, batch_size, model = None, l2_regularization_const = None):
        #differs from calculate_cross_entropy in that it also allows for regularization term
        #### THIS FUNCTION DOES NOT WORK!!!!!
        if model == None:
            model = self.model
        if l2_regularization_const == None:
            l2_regularization_const = self.l2_regularization_const
        excluded_keys = {'bias':['0'], 'weights':[]}
        outputs = self.flatten_output(self.forward_pass(inputs, feature_sequence_lens, model = model), feature_sequence_lens)
        if self.l2_regularization_const == 0.0:
            return self.calculate_cross_entropy(outputs, labels)
        else:
            return self.calculate_cross_entropy(outputs, labels) + (model.norm(excluded_keys) ** 2) * l2_regularization_const / 2. * batch_size
        
    def calculate_classification_statistics(self, features, flat_labels, feature_sequence_lens, model=None):
        if model == None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': []}
        
        if self.do_backprop == False:
            classification_batch_size = 64
        else:
            classification_batch_size = max(self.backprop_batch_size, 64)
        
        batch_index = 0
        end_index = 0
        cross_entropy = 0.0
        log_perplexity = 0.0
        num_correct = 0
        num_sequences = features.shape[1]
        num_examples = self.batch_size(feature_sequence_lens)
#        print features.shape
        while end_index < num_sequences: #run through the batches
            per_done = float(batch_index)/num_sequences*100
            sys.stdout.write("\r                                                                \r") #clear line
            sys.stdout.write("\rCalculating Classification Statistics: %.1f%% done " % per_done), sys.stdout.flush()
            end_index = min(batch_index+classification_batch_size, num_sequences)
            max_seq_len = max(feature_sequence_lens[batch_index:end_index])
#            print batch_index, max_seq_len
            output = self.flatten_output(self.forward_pass(features[:max_seq_len,batch_index:end_index], feature_sequence_lens[batch_index:end_index], model=model), 
                                         feature_sequence_lens[batch_index:end_index])
            start_frame = np.where(flat_labels[:,0] == batch_index)[0][0]
            end_frame = np.where(flat_labels[:,0] == end_index-1)[0][-1] + 1
            label = flat_labels[start_frame:end_frame]
            cross_entropy += self.calculate_cross_entropy(output, label)
            log_perplexity += self.calculate_log_perplexity(output, label)
            #don't use calculate_classification_accuracy() because of possible rounding error
            prediction = output.argmax(axis=1)
            num_correct += np.sum(prediction == label[:,1]) #- (prediction.size - num_examples) #because of the way we handle features, where some observations are null, we want to remove those examples for calculating accuracy
            batch_index += classification_batch_size
        
        sys.stdout.write("\r                                                                \r") #clear line
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        
#        cross_entropy /= np.log(2) * num_examples
        loss /= np.log(2) * num_examples
        log_perplexity /= num_examples
        perplexity = 2 ** log_perplexity
        return cross_entropy, perplexity, num_correct, num_examples, loss
    
    def backprop_mikolov_steepest_descent(self, bptt = 4, bptt_block = 4, independent = True):
        print "Starting backprop using Mikolov steepest descent"
        hidden_hidden_gradient = np.zeros(self.model.weights['hidden_hidden'].shape)
#        gradient = RNNLM_Weight()
#        gradient.init_zero_weights(self.model.get_architecture(), False)
        self.model.bias['visible'] *= 0.0
        self.model.bias['hidden'] *= 0.0
        self.model.bias['output'] *= 0.0
        
        self.model.init_hiddens *= 0.0
        self.model.init_hiddens += 1.0
#        if self.validation_feature_file_name is not None:
#            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
#            print "cross-entropy before steepest descent is", cross_entropy
#            print "perplexity before steepest descent is", perplexity
#            if self.l2_regularization_const > 0.0:
#                print "regularized loss is", loss
#            print "number correctly classified is", num_correct, "of", num_examples
#        
        excluded_keys = {'bias':['0'], 'weights':[]}
#        frame_table = np.cumsum(self.feature_sequence_lens)
        num_words = self.model.weights['hidden_output'].shape[1]
#        print num_words
        bptt_hiddens = np.zeros((bptt + bptt_block, self.num_hiddens))
        word_history = -np.ones((bptt + bptt_block,), dtype=np.int)
        bptt_error = np.zeros((bptt + bptt_block, self.num_hiddens))
        cur_hiddens = np.zeros((self.num_hiddens,))
        prev_hiddens = np.ones((self.num_hiddens,))
        hidden_hidden_buffer = np.empty((self.num_hiddens,))
        output_layer = np.zeros((num_words,))
        num_sequences = self.features.shape[1]
        cur_entropy = 0.0
        one_perc_num_sequences = max(num_sequences / 100, 1)
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
#            batch_index = 0
#            end_index = 0
            cur_entropy = 0.0
            counter = 0
            for seq_index in range(num_sequences):
#                prev_hiddens[:] = 1.0
                word_history[:] = -1
                bptt_hiddens *= 0.0
                features = self.features[:self.feature_sequence_lens[seq_index], seq_index]
                if (seq_index + 1) % one_perc_num_sequences == 0:
                    per_done = float(seq_index) / num_sequences * 100
                    print "Finished %.2f%% of training with average entropy: %f" % (per_done, cur_entropy / counter)
                for feature in features:
#                    print self.features[:self.feature_sequence_lens[index],index]
#                    print self.labels[:self.feature_sequence_lens[index]]
#                    print self.labels.shape#[:self.feature_sequence_lens[index],1]
                    label = self.labels[counter,1]
#                    print feature, label
                    word_history[-1] = feature
                    np.dot(prev_hiddens, self.model.weights['hidden_hidden'], out = cur_hiddens) #Tn-1 to Tn 
                    cur_hiddens += self.model.weights['visible_hidden'][feature,:]
                    np.clip(cur_hiddens, -50.0, 50.0, out=cur_hiddens)
                    cur_hiddens[:] = self.sigmoid(cur_hiddens)
                    bptt_hiddens[-1,:] = cur_hiddens
#                    print cur_hiddens.shape
#                    print np.dot(cur_hiddens, self.model.weights['hidden_output']).shape
                    np.dot(cur_hiddens, self.model.weights['hidden_output'], out = output_layer)
                    output_layer += self.model.bias['output'].reshape((output_layer.size,))
                    output_layer -= np.max(output_layer)
                    np.exp(output_layer, out = output_layer)
                    output_layer /= np.sum(output_layer)
                    cur_entropy -= np.log2(output_layer[label])
                    
                    output_layer[label] -= 1.0
                    self.model.bias['output'] -= self.steepest_learning_rate[epoch_num] * output_layer.reshape((output_layer.size,))
                    self.model.weights['hidden_output'] -= self.steepest_learning_rate[epoch_num] * np.outer(cur_hiddens, output_layer)
                    
                    bptt_error[-1,:] = np.dot(output_layer, self.model.weights['hidden_output'].T)
                    bptt_error[-1,:] *= cur_hiddens
                    bptt_error[-1,:] *= (1 - cur_hiddens)
                    
                    if (counter+1) % bptt_block == 0: #or label == (num_words-1): #encounters end of sentence
#                        print bptt_error
#                        print word_history
                        for time_index in range(bptt+bptt_block-1):
                            word_hist = word_history[-time_index-1]
#                            print time_index, word_hist, word_history
                            if word_hist == -1:
                                break
#                            print time_index
#                            print word_hist
#                            print self.model.weights['visible_hidden'].shape
#                            gradient.weights['visible_hidden'][word_hist,:] += bptt_error[-time_index-1,:]
#                            gradient.weights['hidden_hidden'] += np.outer(bptt_hiddens[-time_index-2,:], bptt_error[-time_index-1,:])
                            self.model.weights['visible_hidden'][word_hist,:] -= self.steepest_learning_rate[epoch_num] * bptt_error[-time_index-1,:]
                            hidden_hidden_gradient += np.outer(bptt_hiddens[-time_index-2,:], bptt_error[-time_index-1,:])
#                            self.model.weights['hidden_hidden'] -= self.steepest_learning_rate[epoch_num] * np.outer(bptt_hiddens[-time_index-2,:], bptt_error[-time_index-1,:])
                            np.dot(bptt_error[-time_index-1,:], self.model.weights['hidden_hidden'].T, out = hidden_hidden_buffer)
                            
                            #now BPTT a step
                            hidden_hidden_buffer *= bptt_hiddens[-time_index-2,:]
                            hidden_hidden_buffer *= (1 - bptt_hiddens[-time_index-2,:])
                            bptt_error[time_index-2,:] += hidden_hidden_buffer
                        
#                        gradient.weights['visible_hidden'][word_history[0],:] += bptt_error[0,:]
                        self.model.weights['visible_hidden'][word_history[0],:] -= self.steepest_learning_rate[epoch_num] * bptt_error[0,:]
                        
#                        print gradient.weights['hidden_hidden']
                        word_history[:] = -1
                        bptt_error *= 0.0
                        self.model.weights['hidden_hidden'] -= self.steepest_learning_rate[epoch_num] * hidden_hidden_gradient
#                        self.model -= gradient * self.steepest_learning_rate[epoch_num]
#                        gradient *= 0.0
#                        if label == (num_words -1):
#                            bptt_hiddens *= 0.0
#                            word_history[:] = -1
                    
                    prev_hiddens[:] = bptt_hiddens[-1,:]
                    bptt_hiddens[:-1,:] = bptt_hiddens[1:,:]
                    bptt_error[:-1,:] = bptt_error[1:,:]
                    word_history[:-1] = word_history[1:]
                    counter += 1
                    
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.feature_sequence_lens, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity before steepest descent is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
                    
                    
                    
#            while end_index < self.num_sequences: #run through the batches
#                per_done = float(batch_index)/self.num_sequences*100
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
#                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
#                max_seq_len = max(self.feature_sequence_lens[batch_index:end_index])
#                batch_inputs = self.features[:max_seq_len,batch_index:end_index]
#                start_frame = np.where(self.labels[:,0] == batch_index)[0][0]
#                end_frame = np.where(self.labels[:,0] == end_index-1)[0][-1] + 1
#                batch_labels = copy.deepcopy(self.labels[start_frame:end_frame,:])
#                batch_labels[:,0] -= batch_labels[0,0]
#                batch_fsl = self.feature_sequence_lens[batch_index:end_index]
#                
##                sys.stdout.write("\r                                                                \r") #clear line
##                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
#                gradient = self.calculate_gradient(batch_inputs, batch_labels, batch_fsl, model=self.model, check_gradient = False)
#                self.model -= gradient * self.steepest_learning_rate[epoch_num]
#                del batch_labels
#                batch_index += self.backprop_batch_size
     
    def backprop_steepest_descent_single_batch(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture())
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture())
        if self.validation_feature_file_name is not None:
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            print "perplexity is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
        
#        excluded_keys = {'bias':['0'], 'weights':[]}
#        frame_table = np.cumsum(self.feature_sequence_lens)
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
                batch_features = self.features[:feature_sequence_len, batch_index]
                batch_labels = self.labels[start_frame:end_frame,1]
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_labels, gradient, return_cross_entropy = True, 
                                                                check_gradient = False)
#                print self.model.norm()
#                print gradient.norm()
                cross_entropy += cur_xent 
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                ppp = cross_entropy / end_frame
                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
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
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))           
            print "Epoch finished at", datetime.datetime.now()
        end_time = datetime.datetime.now()
        print "Training finished at", end_time, "and ran for", end_time - start_time
        
    def backprop_steepest_descent(self):
        print "Starting backprop using steepest descent"
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture())
        if self.validation_feature_file_name is not None:
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            print "perplexity is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
        
#        excluded_keys = {'bias':['0'], 'weights':[]}
#        frame_table = np.cumsum(self.feature_sequence_lens)
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            batch_index = 0
            end_index = 0
            cross_entropy = 0.0
            num_examples = 0
            if hasattr(self, 'momentum_rate'):
                momentum_rate = self.momentum_rate[epoch_num]
                print "momentum is", momentum_rate
            else:
                momentum_rate = 0.0
            while end_index < self.num_sequences: #run through the batches
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                if num_examples > 0:
                    ppp = cross_entropy / num_examples
                    sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
                max_seq_len = max(self.feature_sequence_lens[batch_index:end_index])
                batch_inputs = self.features[:max_seq_len,batch_index:end_index]
                start_frame = np.where(self.labels[:,0] == batch_index)[0][0]
                end_frame = np.where(self.labels[:,0] == end_index-1)[0][-1] + 1
                batch_labels = copy.deepcopy(self.labels[start_frame:end_frame,:])
                batch_labels[:,0] -= batch_labels[0,0]
                batch_fsl = self.feature_sequence_lens[batch_index:end_index]
                batch_size = self.batch_size(self.feature_sequence_lens[batch_index:end_index])
                num_examples += batch_size
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient, cur_xent = self.calculate_gradient(batch_inputs, batch_labels, batch_fsl, model=self.model, check_gradient = False, return_cross_entropy = True)
#                print np.max(np.abs(gradient.weights['hidden_output']))
                cross_entropy += cur_xent
                if self.l2_regularization_const > 0.0:
                    self.model *= (1-self.l2_regularization_const) #l2 regularization_const
                self.model -= gradient * self.steepest_learning_rate[epoch_num] #/ batch_size
                if momentum_rate > 0.0:
                    self.model += prev_step * momentum_rate
                prev_step.assign_weights(gradient)
                prev_step *= -self.steepest_learning_rate[epoch_num] #/ batch_size
                del batch_labels
                batch_index += self.backprop_batch_size
                
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))

    def backprop_adagrad_single_batch(self):
        print "Starting backprop using adagrad"
        adagrad_weight = RNNLM_Weight()
        adagrad_weight.init_zero_weights(self.model.get_architecture())
        
        buffer_weight = RNNLM_Weight()
        buffer_weight.init_zero_weights(self.model.get_architecture())
        
        fudge_factor = 1.0
        adagrad_weight = adagrad_weight + fudge_factor
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture())
#        if self.validation_feature_file_name is not None:
#            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
#            print "cross-entropy before steepest descent is", cross_entropy
#            print "perplexity is", perplexity
#            if self.l2_regularization_const > 0.0:
#                print "regularized loss is", loss
#            print "number correctly classified is", num_correct, "of", num_examples
        
#        excluded_keys = {'bias':['0'], 'weights':[]}
#        frame_table = np.cumsum(self.feature_sequence_lens)
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            start_frame = 0
            end_frame = 0
            cross_entropy = 0.0
            num_examples = 0
#            if hasattr(self, 'momentum_rate'):
#                momentum_rate = self.momentum_rate[epoch_num]
#                print "momentum is", momentum_rate
#            else:
#                momentum_rate = 0.0
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                batch_features = self.features[:feature_sequence_len, batch_index]
                batch_labels = self.labels[start_frame:end_frame,1]
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_labels, gradient, return_cross_entropy = True, 
                                                                check_gradient = False)
#                print self.model.norm()
#                print gradient.norm()
                if self.l2_regularization_const > 0.0:
                    buffer_weight.assign_weights(self.model)
                    buffer_weight *= self.l2_regularization_const
                    gradient += buffer_weight
                buffer_weight.assign_weights(gradient)
#                print gradient.init_hiddens
                buffer_weight **= 2.0
                adagrad_weight += buffer_weight
#                print adagrad_weight.init_hiddens
                buffer_weight.assign_weights(adagrad_weight)
                buffer_weight **= 0.5
#                print buffer_weight.init_hiddens
                gradient /= buffer_weight
#                print gradient.init_hiddens
                cross_entropy += cur_xent 
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                ppp = cross_entropy / end_frame
                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                gradient *= -self.steepest_learning_rate[epoch_num]
                self.model += gradient #/ batch_size
#                if momentum_rate > 0.0:
#                    prev_step *= momentum_rate
#                    self.model += prev_step
#                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num]
                
                start_frame = end_frame
                
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))  

    def backprop_adagrad(self):
        print "Starting backprop using adagrad"
        adagrad_weight = RNNLM_Weight()
        adagrad_weight.init_zero_weights(self.model.get_architecture())
        fudge_factor = 1.0
        adagrad_weight = adagrad_weight + fudge_factor
        if self.validation_feature_file_name is not None:
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before adagrad is", cross_entropy
            print "perplexity is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
        
#        excluded_keys = {'bias':['0'], 'weights':[]}
#        frame_table = np.cumsum(self.feature_sequence_lens)
#        first_batch = True
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            batch_index = 0
            end_index = 0
            cross_entropy = 0.0
            num_examples = 0
#            if hasattr(self, 'momentum_rate'):
#                momentum_rate = self.momentum_rate[epoch_num]
#                print "momentum is", momentum_rate
#            else:
#                momentum_rate = 0.0
            
            while end_index < self.num_sequences: #run through the batches
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                if num_examples > 0:
                    ppp = cross_entropy / num_examples
                    sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
                max_seq_len = max(self.feature_sequence_lens[batch_index:end_index])
                batch_inputs = self.features[:max_seq_len,batch_index:end_index]
                start_frame = np.where(self.labels[:,0] == batch_index)[0][0]
                end_frame = np.where(self.labels[:,0] == end_index-1)[0][-1] + 1
                batch_labels = copy.deepcopy(self.labels[start_frame:end_frame,:])
                batch_labels[:,0] -= batch_labels[0,0]
                batch_fsl = self.feature_sequence_lens[batch_index:end_index]
                batch_size = self.batch_size(self.feature_sequence_lens[batch_index:end_index])
                num_examples += batch_size
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient, cur_xent = self.calculate_gradient(batch_inputs, batch_labels, batch_fsl, model=self.model, check_gradient = False, return_cross_entropy = True)
                
                cross_entropy += cur_xent
                
                if self.l2_regularization_const > 0.0:
                    gradient += (self.model * self.l2_regularization_const) #l2 regularization_const
                    
                adagrad_weight += gradient ** 2
                self.model -= (gradient / (adagrad_weight ** 0.5)) * self.steepest_learning_rate[epoch_num] #/ batch_size
#                if first_batch:
##                    print "normal gradient"
#                    self.model -= gradient * self.steepest_learning_rate[epoch_num]
#                    first_batch = False
#                else:
##                    print "adagrad"
#                    self.model -= (gradient / adagrad_weight) * self.steepest_learning_rate[epoch_num] #/ batch_size
                
#                print adagrad_weight.min()
#                if momentum_rate > 0.0:
#                    self.model += prev_step * momentum_rate
#                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num] #/ batch_size
                del batch_labels
                batch_index += self.backprop_batch_size
                
            if self.validation_feature_file_name is not None:
                cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                print "perplexity is", perplexity
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))


    def pearlmutter_forward_pass(self, inputs, unflattened_labels, feature_sequence_lens, direction, batch_size, hiddens=None, outputs=None, model=None, check_gradient=False, stop_at='output'): #need to test
        """let f be a function from inputs to outputs
        consider the weights to be a vector w of parameters to be optimized, (and direction d to be the same)
        pearlmutter_forward_pass calculates d' \jacobian_w f
        stop_at is either 'linear', 'output', or 'loss' """
        
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(inputs, feature_sequence_lens, model, return_hiddens=True)
            
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_hiddens = architecture[1]
        num_sequences = inputs.shape[1]
        num_outs = architecture[2]
        hidden_deriv = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        output_deriv = np.zeros((max_sequence_observations, num_sequences, num_outs))
#        if stop_at == 'loss':
#            loss_deriv = np.zeros(output_deriv.shape)
        
        #propagate hiddens
#        print model.init_hiddens.shape
        
        hidden_deriv[0,:,:] = (self.forward_layer(inputs[0,:], direction.weights['visible_hidden'], direction.bias['hidden'], 
                                                  model.weight_type['visible_hidden'], prev_hiddens=model.init_hiddens, 
                                                  hidden_hidden_weights=direction.weights['hidden_hidden']) 
                               + np.dot(direction.init_hiddens, model.weights['hidden_hidden'])) * hiddens[0,:,:] * (1 - hiddens[0,:,:])
        linear_layer = (self.weight_matrix_multiply(hiddens[0,:,:], direction.weights['hidden_output'], 
                                                    direction.bias['output']) +
                        np.dot(hidden_deriv[0,:,:], model.weights['hidden_output']))
        if stop_at == 'linear':
            output_deriv[0,:,:] = linear_layer
        elif stop_at == 'output':
            output_deriv[0,:,:] = linear_layer * outputs[0,:,:] - outputs[0,:,:] * np.sum(linear_layer * outputs[0,:,:], axis=1)[:,np.newaxis]
#        if stop_at == 'loss':
#            output_deriv[model.num_layers+1] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:]
            hidden_deriv[sequence_index,:,:] = (self.forward_layer(sequence_input, direction.weights['visible_hidden'], direction.bias['hidden'], 
                                                                   model.weight_type['visible_hidden'], prev_hiddens=model.init_hiddens, 
                                                                   hidden_hidden_weights=direction.weights['hidden_hidden'])  
                                                + np.dot(hidden_deriv[sequence_index-1,:,:], model.weights['hidden_hidden'])) * hiddens[sequence_index,:,:] * (1 - hiddens[sequence_index,:,:])
            linear_layer = (self.weight_matrix_multiply(hiddens[sequence_index,:,:], direction.weights['hidden_output'], 
                                                        direction.bias['output']) +
                            np.dot(hidden_deriv[sequence_index,:,:], model.weights['hidden_output']))
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(feature_sequence_lens <= sequence_index)
            hidden_deriv[sequence_index,zero_input,:] = 0.0
            output_deriv[sequence_index,zero_input,:] = 0.0
            if stop_at == 'linear':
                output_deriv[sequence_index,:,:] = linear_layer
            else:
                output_deriv[sequence_index,:,:] = linear_layer * outputs[sequence_index,:,:] - outputs[sequence_index,:,:] * np.sum(linear_layer * outputs[sequence_index,:,:], axis=1)[:,np.newaxis]
#            if stop_at == 'loss':
#                loss_deriv[sequence_index,:,:] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
        if not check_gradient:
            return output_deriv, hidden_deriv
        #compare with finite differences approximation
        else:
            epsilon = 1E-5
            if stop_at == 'linear':
                calculated = output_deriv
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon, linear_output=True)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon, linear_output=True)
            elif stop_at == 'output':
                calculated = output_deriv
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon)
#            elif stop_at == 'loss':
#                calculated = hidden_deriv[model.num_layers + 1]
#                finite_diff_forward = -np.log([max(self.forward_pass(inputs, model = model + direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
#                finite_diff_backward =  -np.log([max(self.forward_pass(inputs, model = model - direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
            for seq in range(num_sequences):
                finite_diff_approximation = ((finite_diff_forward - finite_diff_backward) / (2 * epsilon))[:,seq,:]
                print "At sequence", seq
                print "pearlmutter calculation"
                print calculated[:,seq,:]
                print "finite differences approximation, epsilon", epsilon
                print finite_diff_approximation
            sys.exit()
            
    def calculate_per_example_cross_entropy(self, example_output, example_label):
        if example_label.size > 1:
            return -np.sum(np.log(np.clip(example_output, a_min=1E-12, a_max=1.0)) * example_label)
        else:
            return -np.log(np.clip(example_output[example_label], a_min=1E-12, a_max=1.0))
        
    def calculate_second_order_direction(self, inputs, unflattened_labels, feature_sequence_lens, batch_size, direction = None, model = None, second_order_type = None, 
                                         hiddens = None, outputs=None, check_direction = False, structural_damping_const = 0.0): #need to test
        #given an input direction direction, the function returns H*d, where H is the Hessian of the weight vector
        #the function does this efficient by using the Pearlmutter (1994) trick
        excluded_keys = {'bias': ['0'], 'weights': []}
        if model == None:
            model = self.model
        if direction == None:
            direction = self.calculate_gradient(inputs, unflattened_labels, feature_sequence_lens, check_gradient = False, model = model)
        if second_order_type == None:
            second_order_type='gauss-newton' #other option is 'hessian'
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(inputs, feature_sequence_lens, model = model, return_hiddens=True)   
        
        
        if second_order_type == 'gauss-newton':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, feature_sequence_lens, direction, batch_size, hiddens, outputs, model, stop_at='output') #nbatch x nout
            second_order_direction = self.backward_pass(output_deriv, hiddens, inputs, model, structural_damping_const, hidden_deriv)
        elif second_order_type == 'hessian':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, feature_sequence_lens, direction, batch_size, hiddens, outputs, model, stop_at='output') #nbatch x nout
            second_order_direction = self.pearlmutter_backward_pass(hidden_deriv, unflattened_labels, hiddens, model, direction)
        elif second_order_type == 'fisher':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, feature_sequence_lens, direction, batch_size, hiddens, outputs, model, stop_at='loss')#nbatch x nout
            weight_vec = output_deriv - unflattened_labels
            weight_vec *= hidden_deriv[model.num_layers+1][:, np.newaxis] #TODO: fix this line
            second_order_direction = self.backward_pass(weight_vec, hiddens, inputs, model) 
        else:
            print second_order_type, "is not a valid type. Acceptable types are gauss-newton, hessian, and fisher... Exiting now..."
            sys.exit()
            
        if not check_direction:
            if self.l2_regularization_const > 0.0:
                return second_order_direction / batch_size + direction * self.l2_regularization_const
            return second_order_direction / batch_size
        
        ##### check direction only if you think there is a problem #######
        else:
            finite_difference_model = RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
            
            if second_order_type == 'gauss-newton':
                #assume that pearlmutter forward pass is correct because the function has a check_gradient flag to see if it's is
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Gv\n"), sys.stdout.flush()
                linear_out = self.forward_pass(inputs, model = model, linear_output=True)
                num_examples = self.batch_size(feature_sequence_lens)
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon, linear_output=True)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon, linear_output=True)
                finite_diff_jacobian_vec = (finite_diff_forward - finite_diff_backward) / (2 * epsilon)
                flat_finite_diff_jacobian_vec = self.flatten_output(finite_diff_jacobian_vec, feature_sequence_lens)
                flat_linear_out = self.flatten_output(linear_out, feature_sequence_lens)
                flat_labels = self.flatten_output(unflattened_labels, feature_sequence_lens)
                
                flat_finite_diff_HJv = np.zeros(flat_finite_diff_jacobian_vec.shape)
                
                num_outputs = flat_linear_out.shape[1]
                collapsed_hessian = np.zeros((num_outputs,num_outputs))
                for example_index in range(num_examples):
                    #calculate collapsed Hessian
                    direction1 = np.zeros(num_outputs)
                    direction2 = np.zeros(num_outputs)
                    for index1 in range(num_outputs):
                        for index2 in range(num_outputs):
                            direction1[index1] = epsilon
                            direction2[index2] = epsilon
                            example_label = np.array(flat_labels[example_index])
                            loss_plus_plus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] + direction1 + direction2])), example_label)
                            loss_plus_minus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] + direction1 - direction2])), example_label)
                            loss_minus_plus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] - direction1 + direction2])), example_label)
                            loss_minus_minus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] - direction1 - direction2])), example_label)
                            collapsed_hessian[index1,index2] = (loss_plus_plus + loss_minus_minus - loss_minus_plus - loss_plus_minus) / (4 * epsilon * epsilon)
                            direction1[index1] = 0.0
                            direction2[index2] = 0.0
#                    print collapsed_hessian
                    out = self.softmax(flat_linear_out[example_index:example_index+1])
#                    print np.diag(out[0]) - np.outer(out[0], out[0])
                    flat_finite_diff_HJv[example_index] += np.dot(collapsed_hessian, flat_finite_diff_jacobian_vec[example_index])
                    
                obs_so_far = 0
                for sequence_index, num_obs in enumerate(feature_sequence_lens):
                    print "at sequence index", sequence_index
                    #calculate J'd = J'HJv
                    update = RNNLM_Weight()
                    update.init_zero_weights(self.model.get_architecture(), verbose=False)
                    for index in range(direction.init_hiddens.size):
                        update.init_hiddens[0][index] = epsilon
                        #print direction.norm()
                        forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model + update, linear_output=True)
                        backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model - update, linear_output=True)
                        for obs_index in range(num_obs):
                            example_index = obs_so_far + obs_index
                            finite_difference_model.init_hiddens[0][index] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                     flat_finite_diff_HJv[example_index])
                            update.init_hiddens[0][index] = 0.0
                    for key in direction.bias.keys():
                        print "at bias key", key
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            #print direction.norm()
                            forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model + update, linear_output=True)
                            backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model - update, linear_output=True)
                            for obs_index in range(num_obs):
                                example_index = obs_so_far + obs_index
                                finite_difference_model.bias[key][0][index] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                      flat_finite_diff_HJv[example_index])
                            update.bias[key][0][index] = 0.0
                    for key in direction.weights.keys():
                        print "at weight key", key
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model= model + update, linear_output=True)
                                backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model= model - update, linear_output=True)
                                for obs_index in range(num_obs):
                                    example_index = obs_so_far + obs_index
                                    finite_difference_model.weights[key][index0][index1] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                                   flat_finite_diff_HJv[example_index])
                                update.weights[key][index0][index1] = 0.0
                    obs_so_far += num_obs
            elif second_order_type == 'hessian':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Hv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    update = RNNLM_Weight()
                    update.init_zero_weights(self.model.get_architecture(), verbose=False)
                    
                    current_gradient = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, model=model, l2_regularization_const = 0.)
                    
                    for key in finite_difference_model.bias.keys():
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            forward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                   model = model + update, l2_regularization_const = 0.)
                            backward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                    model = model - update, l2_regularization_const = 0.)
                            finite_difference_model.bias[key][0][index] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                            update.bias[key][0][index] = 0.0
        
                    for key in finite_difference_model.weights.keys():
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                       model = model + update, l2_regularization_const = 0.) 
                                backward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                        model = model - update, l2_regularization_const = 0.)
                                finite_difference_model.weights[key][index0][index1] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                                update.weights[key][index0][index1] = 0.0
            elif second_order_type == 'fisher':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Fv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    current_gradient = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, model = model, l2_regularization_const = 0.)                
                    finite_difference_model += current_gradient * current_gradient.dot(direction, excluded_keys)
            
            print "calculated second order direction for init hiddens"
            print second_order_direction.init_hiddens
            print "finite difference approximation for init hiddens"
            print finite_difference_model.init_hiddens
            
            for bias_cur_layer in direction.bias.keys():
                print "calculated second order direction for bias", bias_cur_layer
                print second_order_direction.bias[bias_cur_layer]
                print "finite difference approximation for bias", bias_cur_layer
                print finite_difference_model.bias[bias_cur_layer]
            for weight_cur_layer in finite_difference_model.weights.keys():
                print "calculated second order direction for weights", weight_cur_layer
                print second_order_direction.weights[weight_cur_layer]
                print "finite difference approximation for weights", weight_cur_layer
                print finite_difference_model.weights[weight_cur_layer]
            sys.exit()
        ##########################################################
        
    def backprop_truncated_newton(self):
        print "Starting backprop using truncated newton"
        
#        cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.feature_sequence_lens, self.model)
#        print "cross-entropy before steepest descent is", cross_entropy
#        print "perplexity before steepest descent is", perplexity
#        if self.l2_regularization_const > 0.0:
#            print "regularized loss is", loss
#        print "number correctly classified is", num_correct, "of", num_examples
        
        excluded_keys = {'bias':['0'], 'weights':[]} 
        damping_factor = self.truncated_newton_init_damping_factor
        preconditioner = None
        model_update = None
        cur_done = 0.0
        
        for epoch_num in range(self.num_epochs):
            print "Epoch", epoch_num+1, "of", self.num_epochs
            batch_index = 0
            end_index = 0
            
            while end_index < self.num_sequences: #run through the batches
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
#                if per_done > cur_done + 1.0:
#                    cur_done = per_done
#                    cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.feature_sequence_lens, self.model)
#                    print "cross-entropy before steepest descent is", cross_entropy
#                    print "perplexity before steepest descent is", perplexity
#                    if self.l2_regularization_const > 0.0:
#                        print "regularized loss is", loss
#                    print "number correctly classified is", num_correct, "of", num_examples
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\rdamping factor is %f\r" % damping_factor), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
                max_seq_len = max(self.feature_sequence_lens[batch_index:end_index])
                batch_inputs = self.features[:max_seq_len,batch_index:end_index]
                start_frame = np.where(self.labels[:,0] == batch_index)[0][0]
                end_frame = np.where(self.labels[:,0] == end_index-1)[0][-1] + 1
                batch_unflattened_labels = copy.deepcopy(self.labels[start_frame:end_frame,:])
                batch_unflattened_labels[:,0] -= batch_unflattened_labels[0,0]
                batch_fsl = self.feature_sequence_lens[batch_index:end_index]
#                batch_inputs = self.features[:,batch_index:end_index]
#                batch_unflattened_labels = self.unflattened_labels[:,batch_index:end_index,:]
                batch_size = self.batch_size(self.feature_sequence_lens[batch_index:end_index])
                
#                sys.stdout.write("\r                                                                \r") #clear line
#                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient = self.calculate_gradient(batch_inputs, batch_unflattened_labels, batch_fsl, model=self.model, check_gradient = False)
                
                old_loss = self.calculate_loss(batch_inputs, batch_fsl, batch_unflattened_labels, batch_size, model=self.model) 
                
                if False: #self.use_fisher_preconditioner:
                    sys.stdout.write("\r                                                                \r")
                    sys.stdout.write("calculating diagonal Fisher matrix for preconditioner"), sys.stdout.flush()
                    
                    preconditioner = self.calculate_fisher_diag_matrix(batch_inputs, batch_unflattened_labels, False, self.model, l2_regularization_const = 0.0)
                    # add regularization
                    #preconditioner = preconditioner + alpha / preconditioner.size(excluded_keys) * self.model.norm(excluded_keys) ** 2
                    preconditioner = (preconditioner + self.l2_regularization_const + damping_factor) ** (3./4.)
                    preconditioner = preconditioner.clip(preconditioner.max(excluded_keys) * self.fisher_preconditioner_floor_val, float("Inf"))
                model_update, model_vals = self.conjugate_gradient(batch_inputs, batch_unflattened_labels, batch_fsl, batch_size, self.truncated_newton_num_cg_epochs, 
                                                                   model=self.model, damping_factor=damping_factor, preconditioner=preconditioner, 
                                                                   gradient=gradient, second_order_type=self.second_order_matrix, 
                                                                   init_search_direction=None, verbose = False,
                                                                   structural_damping_const = self.structural_damping_const)
                model_den = model_vals[-1] #- model_vals[0]
                
                self.model += model_update
                new_loss = self.calculate_loss(batch_inputs, batch_fsl, batch_unflattened_labels, batch_size, model=self.model) 
                model_num = (new_loss - old_loss) / batch_size
#                sys.stdout.write("\r                                                                      \r") #clear line
#                print "model ratio is", model_num / model_den,
                if model_num / model_den < 0.25:
                    damping_factor *= 1.5
                elif model_num / model_den > 0.75:
                    damping_factor *= 2./3.
                batch_index += self.backprop_batch_size
                
            cross_entropy, perplexity, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.feature_sequence_lens, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            print "perplexity before steepest descent is", perplexity
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
                
    def conjugate_gradient(self, batch_inputs, batch_unflattened_labels, batch_feature_sequence_lens, batch_size, num_epochs, model = None, damping_factor = 0.0, #seems to be correct, compare with conjugate_gradient.py
                           verbose = False, preconditioner = None, gradient = None, second_order_type='gauss-newton', 
                           init_search_direction = None, structural_damping_const = 0.0):
        """minimizes function q_x(p) = \grad_x f(x)' p + 1/2 * p'Gp (where x is fixed) use linear conjugate gradient"""
        if verbose:
            print "preconditioner is", preconditioner
        excluded_keys = {'bias':['0'], 'weights':[]} 
        if model == None:
            model = self.model
        
        tolerance = 5E-4
        gap_ratio = 0.1
        min_gap = 10
        #max_test_gap = int(np.max([np.ceil(gap_ratio * num_epochs), min_gap]) + 1)
        model_vals = list()
        
        model_update = RNNLM_Weight()
        model_update.init_zero_weights(model.get_architecture())
        
        outputs, hiddens = self.forward_pass(batch_inputs, model, return_hiddens=True)
        if gradient == None:
            gradient = self.calculate_gradient(batch_inputs, batch_unflattened_labels, batch_feature_sequence_lens, batch_size, model = model, hiddens = hiddens, outputs = outputs)
        
        if init_search_direction == None:
            model_vals.append(0)
            residual = gradient 
        else:
            second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, batch_feature_sequence_lens, batch_size, init_search_direction, 
                                                                           model, second_order_type=second_order_type, hiddens = hiddens,
                                                                           structural_damping_const = structural_damping_const * damping_factor)
            residual = gradient + second_order_direction
            model_val = 0.5 * init_search_direction.dot(gradient + residual, excluded_keys)
            model_vals.append(model_val) 
            model_update += init_search_direction    
            
        if verbose:
            print "model val at end of epoch is", model_vals[-1]
        
        if preconditioner != None:
            preconditioned_residual = residual / preconditioner
        else:
            preconditioned_residual = residual
        search_direction = -preconditioned_residual
        residual_dot = residual.dot(preconditioned_residual, excluded_keys)
        for epoch in range(num_epochs):
#            print "\r                                                                \r", #clear line
#            sys.stdout.write("\rconjugate gradient epoch %d of %d\r" % (epoch+1, num_epochs)), sys.stdout.flush()
            
            if damping_factor > 0.0:
                #TODO: check to see if ... + search_direction * damping_factor is correct with structural damping
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, batch_feature_sequence_lens, batch_size, search_direction, model, second_order_type=second_order_type, hiddens = hiddens, 
                                                                               structural_damping_const = damping_factor * structural_damping_const) + search_direction * damping_factor
            else:
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, batch_feature_sequence_lens, batch_size, search_direction, model, second_order_type=second_order_type, hiddens = hiddens)
                                                                            
            curvature = search_direction.dot(second_order_direction,excluded_keys)
            if curvature <= 0:
                print "curvature must be positive, but is instead", curvature, "returning current weights"
                break
            
            step_size = residual_dot / curvature
            if verbose:
                print "residual dot search direction is", residual.dot(search_direction, excluded_keys)
                print "residual dot is", residual_dot
                print "curvature is", curvature
                print "step size is", step_size
            model_update += search_direction * step_size
            
            residual += second_order_direction * step_size
            model_val = 0.5 * model_update.dot(gradient + residual, excluded_keys)
            model_vals.append(model_val)
            if verbose:
                print "model val at end of epoch is", model_vals[-1]
            test_gap = int(np.max([np.ceil(epoch * gap_ratio), min_gap]))
            if epoch > test_gap: #checking termination condition
                previous_model_val = model_vals[-test_gap]
                if (previous_model_val - model_val) / model_val <= tolerance * test_gap and previous_model_val < 0:
                    print "\r                                                                \r", #clear line
                    sys.stdout.write("\rtermination condition satisfied for conjugate gradient, returning step\r"), sys.stdout.flush()
                    break
            if preconditioner != None:
                preconditioned_residual = residual / preconditioner
            else:
                preconditioned_residual = residual
            new_residual_dot = residual.dot(preconditioned_residual, excluded_keys)
            conjugate_gradient_const = new_residual_dot / residual_dot
            search_direction = -preconditioned_residual + search_direction * conjugate_gradient_const
            residual_dot = new_residual_dot
        return model_update, model_vals
   
    def unflatten_labels(self, labels, sentence_ids):
        num_frames_per_sentence = np.bincount(sentence_ids)
        num_outs = len(np.unique(labels))
        max_num_frames_per_sentence = np.max(num_frames_per_sentence)
        unflattened_labels = np.zeros((max_num_frames_per_sentence, np.max(sentence_ids) + 1, num_outs)) #add one because first sentence starts at 0
        current_sentence_id = 0
        observation_index = 0
        for label, sentence_id in zip(labels,sentence_ids):
            if sentence_id != current_sentence_id:
                current_sentence_id = sentence_id
                observation_index = 0
            unflattened_labels[observation_index, sentence_id, label] = 1.0
            observation_index += 1
        return unflattened_labels
    