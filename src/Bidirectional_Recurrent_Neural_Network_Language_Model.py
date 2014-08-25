'''
Created on Aug 22, 2014

@author: sumanravuri
'''
import sys
import numpy as np
import scipy.io as sp
import copy
from Vector_Math import Vector_Math
from scipy.special import expit
from Bidirectional_RNNLM_Weight import Bidirectional_RNNLM_Weight

class Bidirectional_Recurrent_Neural_Network_Language_Model(object, Vector_Math):
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
        self.model = Bidirectional_RNNLM_Weight()
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

    def forward_layer(self, inputs, weights, biases, weight_type, secondary_inputs = None, secondary_weights = None): #completed
#        raise ValueError("forward_layer() not implemented yet")
        if weight_type == 'logistic':
            return self.softmax(self.weight_matrix_multiply(inputs, weights, biases) + np.dot(secondary_inputs, secondary_weights))
        elif weight_type == 'rbm_gaussian_bernoulli' or weight_type == 'rbm_bernoulli_bernoulli':
            return self.sigmoid(weights[(inputs),:] + self.weight_matrix_multiply(secondary_inputs, secondary_weights, biases))
        #added to test finite differences calculation for pearlmutter forward pass
        elif weight_type == 'linear': #only used for the logistic layer
            return self.weight_matrix_multiply(inputs, weights, biases) + np.dot(secondary_inputs, secondary_weights)
        else:
            print "weight_type", weight_type, "is not a valid layer type.",
            print "Valid layer types are", self.model.valid_layer_types,"Exiting now..."
            sys.exit()
            
    def forward_pass_single_batch(self, inputs, model = None, return_hiddens = False, linear_output = False):
        """forward pass for single batch size. Mainly for speed in this case
        """
        if model == None:
            model = self.model
        num_observations = inputs.size
        hiddens_forward = model.weights['visible_hidden'][(inputs),:]
        hiddens_forward[:1,:] += self.weight_matrix_multiply(model.init_hiddens['forward'], model.weights['hidden_hidden_forward'], model.bias['hidden_forward'])
        expit(hiddens_forward[0,:], hiddens_forward[0,:])
        
        hiddens_backward = model.weights['visible_hidden'][(inputs),:]
        hiddens_backward[-1:,:] += self.weight_matrix_multiply(model.init_hiddens['backward'], model.weights['hidden_hidden_backward'], model.bias['hidden_backward'])
        expit(hiddens_backward[-1,:], hiddens_backward[-1,:])
        
        for time_step in range(1, num_observations):
            hiddens_forward[time_step:time_step+1,:] += self.weight_matrix_multiply(hiddens_forward[time_step-1:time_step,:], 
                                                                                    model.weights['hidden_hidden_forward'], model.bias['hidden_forward'])
            expit(hiddens_forward[time_step,:], hiddens_forward[time_step,:]) #sigmoid
            
            hiddens_backward[num_observations-time_step-1:num_observations-time_step,:] += self.weight_matrix_multiply(hiddens_backward[num_observations-time_step:num_observations-time_step+1,:], 
                                                                                                                       model.weights['hidden_hidden_backward'], model.bias['hidden_backward'])
            expit(hiddens_backward[num_observations-time_step-1,:], hiddens_backward[num_observations-time_step-1,:]) #sigmoid
        
        outputs = self.forward_layer(hiddens_forward, model.weights['hidden_output_forward'], model.bias['output'], model.weight_type['hidden_output'], hiddens_backward, model.weights['hidden_output_backward'])
        
        if return_hiddens:
            return outputs, hiddens_forward, hiddens_backward
        else:
            del hiddens_forward, hiddens_backward
            return outputs
            
    def forward_pass(self, inputs, feature_sequence_lens, model=None, return_hiddens=False, linear_output=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        raise ValueError("forward_pass() not implemented yet")
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_sequences = inputs.shape[1]
        num_hiddens = architecture[1]
        num_outs = architecture[2]
        hiddens_forward = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        hiddens_backward = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        #propagate hiddens
        hiddens_forward[0,:,:] = self.forward_layer(inputs[0,:], model.weights['visible_hidden'], model.bias['hidden'], 
                                                    model.weight_type['visible_hidden'], model.init_hiddens['forward'], 
                                                    model.weights['hidden_hidden_forward'])
        
        hiddens_backward[0,:,:] = self.forward_layer(inputs[0,:], model.weights['visible_hidden'], model.bias['hidden'], 
                                                    model.weight_type['visible_hidden'], model.init_hiddens['backward'], 
                                                    model.weights['hidden_hidden_backward'])
        if linear_output:
            outputs[0,:,:] = self.forward_layer(hiddens_forward[0,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                'linear', )
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