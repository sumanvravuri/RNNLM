'''
Created on Jul 2, 2015

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

class RNNLM_Neural_Network_Addressee(object, Vector_Math):
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
        self.features, self.feature_sequence_lens, self.dssm = self.read_feature_file()
        self.model = RNNLM_Weight()
        self.output_name = self.default_variable_define(config_dictionary, 'output_name', arg_type='string')
        self.single_prediction = self.default_variable_define(config_dictionary, 'single_prediction', arg_type='boolean')
        
        self.required_variables = dict()
        self.all_variables = dict()
        self.required_variables['train'] = ['mode', 'feature_file_name', 'output_name', 'single_prediction']
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'num_hiddens', 'weight_matrix_name', 
#                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 
                               'save_each_epoch', 'backprop_batch_size',
#                               'do_pretrain', 'pretrain_method', 'pretrain_iterations', 
#                               'pretrain_learning_rate', 'pretrain_batch_size',
#                               'do_backprop', 'backprop_method', 'backprop_batch_size', 'l2_regularization_const',
#                               'num_epochs', 'num_line_searches', 'armijo_const', 'wolfe_const',
                               'steepest_learning_rate', 'momentum_rate',
#                               'conjugate_max_iterations', 'conjugate_const_type',
#                               'truncated_newton_num_cg_epochs', 'truncated_newton_init_damping_factor',
#                               'krylov_num_directions', 'krylov_num_batch_splits', 'krylov_num_bfgs_epochs', 'second_order_matrix',
#                               'krylov_use_hessian_preconditioner', 'krylov_eigenvalue_floor_const', 
#                               'fisher_preconditioner_floor_val', 'use_fisher_preconditioner',
#                               'structural_damping_const', 
                               'validation_feature_file_name', 'validation_label_file_name',
                               'use_maxent', 'seed']
        self.required_variables['test'] =  ['mode', 'feature_file_name', 'weight_matrix_name', 'output_name', 'single_prediction']
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
            if 'num_rows' in feature_data:
                num_rows = feature_data['num_rows']
                num_cols = feature_data['num_cols']
                sparse_data = feature_data['data'].astype(np.int32).ravel()
                sparse_indices = feature_data['indices'].astype(np.int32).ravel()
                sparse_indptr = feature_data['indptr'].astype(np.int32).ravel()
                features = ssp.csr_matrix((sparse_data, sparse_indices, sparse_indptr), shape = (num_rows, num_cols))
                sequence_len = feature_data['feature_sequence_lengths']
                sequence_len = np.reshape(sequence_len, (sequence_len.size,))
                dssm = True
            elif 'features' in feature_data:
                feature_data = sp.loadmat(feature_file_name)
                features = feature_data['features'].astype(np.int32)
                sequence_len = feature_data['feature_sequence_lengths'].ravel()
#                sequence_len = np.reshape(sequence_len, (sequence_len.size,))
                dssm = False
            else:
                print "ERROR: feature_data of unknown type"
                raise ValueError
            return features, sequence_len, dssm#in MATLAB format
        except IOError:
            print "Unable to open ", feature_file_name, "... Exiting now"
            raise IOError
    
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

    def make_frame_labels(self, fsl, labels):
        assert(labels.shape[0] == labels.size)
        out_labels = np.zeros((sum(fsl,)))
        start_index = 0
        for idx, seq_len in enumerate(fsl):
            end_index = start_index + seq_len
            out_labels[start_index:end_index] = labels[idx]
            start_index = end_index
        
        return out_labels.astype(np.int32)
    
    def make_dssm_batch_features(self, inputs, fsl):
        if type(inputs) != ssp.csr_matrix:
            raise ValueError('inputs are not of dssm sparse type')
        if fsl.size == 1:
            return inputs
        
        max_obs = max(fsl)
        num_obs = sum(fsl)
        batch_size = fsl.size
        
        cumsum_fsl = np.cumsum(np.hstack((0,fsl[:-1])))
                  
        transform_matrix = np.zeros((batch_size*max_obs, num_obs))
        
        for obs_index in range(max_obs):
            start_index = obs_index * batch_size
            end_index = (obs_index + 1) * batch_size
            out_rows = np.arange(start_index, end_index)
            in_rows = cumsum_fsl + obs_index
            
            out_rows = out_rows[obs_index < fsl]
            in_rows = in_rows[obs_index < fsl]
            
            transform_matrix[out_rows, in_rows] = 1
        
        return ssp.csr_matrix(transform_matrix).dot(inputs)
    
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
            raise ValueError
        if self.labels.shape[0] != self.feature_sequence_lens.size:
            print "Number of examples in feature file: ", self.feature_sequence_lens.size, " does not equal size of label file, ", self.labels.shape[0], "... Exiting now"
            quit()
#        if  [i for i in np.unique(self.labels)] != range(np.max(self.labels)+1):
#            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
#            sys.exit()
#        label_counts = np.bincount(np.ravel(self.labels[:,1])) #[self.labels.count(x) for x in range(np.max(self.labels)+1)]
#        print "distribution of labels is:"
#        for x in range(len(label_counts)):
#            print "#", x, "\b's:", label_counts[x]            
        print "labels seem copacetic"

    def forward_layer(self, inputs, weights, biases, weight_type, prev_hiddens = None, hidden_hidden_weights = None): #completed
        if weight_type == 'logistic':
            if hidden_hidden_weights is None:
                return self.softmax(self.weight_matrix_multiply(inputs, weights, biases))
            else:
                return self.softmax(self.weight_matrix_multiply(inputs, weights, biases) + hidden_hidden_weights[(inputs),:])
        elif weight_type == 'rbm_gaussian_bernoulli' or weight_type == 'rbm_bernoulli_bernoulli':
            return self.sigmoid(inputs.T.dot(weights.T).T + self.weight_matrix_multiply(prev_hiddens, hidden_hidden_weights, biases))
        #added to test finite differences calculation for pearlmutter forward pass
        elif weight_type == 'linear': #only used for the logistic layer
            return self.weight_matrix_multiply(inputs, weights, biases)
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
        hiddens = model.weights['visible_hidden'][(inputs),:]
        hiddens[:1,:] += self.weight_matrix_multiply(model.init_hiddens, model.weights['hidden_hidden'], model.bias['hidden'])
#        np.clip(hiddens[0, :], a_min = 0.0, out = hiddens[0, :])
        expit(hiddens[0,:], hiddens[0,:])
        
        for time_step in range(1, num_observations):
            hiddens[time_step:time_step+1,:] += self.weight_matrix_multiply(hiddens[time_step-1:time_step,:], model.weights['hidden_hidden'], model.bias['hidden'])
#            np.clip(hiddens[time_step, :], a_min = 0.0, out = hiddens[time_step, :])
            expit(hiddens[time_step,:], hiddens[time_step,:]) #sigmoid
        
        if 'visible_output' in model.weights:
            outputs = self.forward_layer(hiddens, model.weights['hidden_output'], model.bias['output'], model.weight_type['hidden_output'],
                                         model.weights['visible_output'])
        else:
            outputs = self.forward_layer(hiddens, model.weights['hidden_output'], model.bias['output'], model.weight_type['hidden_output'])
        
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
            return outputs
    
    
    def forward_pass_multi_batch(self, inputs, feature_sequence_lens, model = None,
                                 return_hiddens = False, single_prediction = True):
        """forward pass for single batch size. Mainly for speed in this case
        single prediction means that only one prediction is given at the end of the RNNLM
        """
        if model == None:
            model = self.model
        
#        if type(inputs) == ssp.csr_matrix:
#            raise ValueError('DSSM not yet implemented')
#        print inputs.shape
        num_observations, batch_size = max(feature_sequence_lens), feature_sequence_lens.size
        mask = np.zeros((batch_size, model.weights['visible_hidden'].shape[1]))
        if not self.dssm:
            linear_feats = inputs.ravel()
            hiddens = model.weights['visible_hidden'][(linear_feats),:]# + model.bias['visible_hidden']
        else:
            hiddens = self.make_dssm_batch_features(inputs, feature_sequence_lens).dot(model.weights['visible_hidden'])
#        if len(inputs.shape) != 1: #means we are using DSSM
#            embedding = inputs.dot(model.weights['visible_hidden']) + model.bias['visible_hidden']
#        else:
#            embedding = model.weights['visible_hidden'][(inputs),:] + model.bias['visible_hidden']
        
#        update_gate = self.weight_matrix_multiply(embedding, model.weights['visible_updategate'], model.bias['updategate'])#hack because of crappy sparse matrix support
#        reset_gate = self.weight_matrix_multiply(embedding, model.weights['visible_resetgate'], model.bias['resetgate'])
#        forget_gate = self.weight_matrix_multiply(embedding, model.weights['visible_forgetgate'], model.bias['forgetgate'])
        hiddens[:batch_size] += self.weight_matrix_multiply(model.init_hiddens, model.weights['hidden_hidden'], 
                                                            model.bias['hidden'])
        
#        hiddens = np.empty(update_gate.shape)
#        cell = np.empty(input_gate.shape)
        #propagate layer at first time step
#        print "linear input_gate"
#        print input_gate[0]
#        expit(update_gate[:batch_size], update_gate[:batch_size])
        expit(hiddens[:batch_size], hiddens[:batch_size])
#        expit(reset_gate[:batch_size], reset_gate[:batch_size])
#        cell[:batch_size] = part_cell[:batch_size] * input_gate[:batch_size]
#        print "input_gate"
#        print input_gate[:batch_size]
#        output_gate[:batch_size] += np.dot(cell[:batch_size], model.weights['curcell_outputgate'])
#        expit(output_gate[:batch_size], output_gate[:batch_size])
#        print "tanh(cell)"
#        print np.tanh(cell[:batch_size])
#        print "output_gate"
#        print output_gate[:batch_size]
#        hiddens[:batch_size] = update_gate[:batch_size] * activation[:batch_size]#output_gate[:batch_size] * np.tanh(cell[:batch_size])
#        print "hiddens"
#        print hiddens[0]
        
        if not single_prediction and not return_hiddens:
            output = np.empty((sum(feature_sequence_lens), model.bias['output'].size))
            zero_out = np.array([0])
            out_fsl = np.cumsum(np.hstack((zero_out, feature_sequence_lens[:-1])))
#            A = np.eye(batch_size)
#            output_buffer = 
            output[out_fsl,:] = self.softmax(self.weight_matrix_multiply(hiddens[:batch_size], 
                                                                         model.weights['hidden_output'], 
                                                                         model.bias['output']))
        elif not single_prediction:
            output_mask = np.zeros((hiddens.shape[0],model.bias['output'].size))
            output_mask[:batch_size,:] = 1
        
        for time_step in range(1, num_observations):
#            print time_step

            mask[:] = 1.0
            mask[(feature_sequence_lens <= time_step), :] = 0.0
            start_index = time_step*batch_size
            end_index = (time_step+1)*batch_size
            prev_index = (time_step-1)*batch_size
            
            hiddens[start_index:end_index] += self.weight_matrix_multiply(hiddens[prev_index:start_index], model.weights['hidden_hidden'], model.bias['hidden'])
#            reset_gate[start_index:end_index] += np.dot(cell[prev_index:start_index], model.weights['prevcell_inputgate'])
            expit(hiddens[start_index:end_index], hiddens[start_index:end_index])
            hiddens[start_index:end_index] *= mask
            
            if not single_prediction and not return_hiddens:
                output_index = out_fsl[feature_sequence_lens > time_step] + time_step
                output[output_index] = self.softmax(self.weight_matrix_multiply(hiddens[start_index:end_index], 
                                                                                model.weights['hidden_output'], 
                                                                                model.bias['output']))[feature_sequence_lens > time_step]
            elif not single_prediction:
#                print start_index, np.where(feature_sequence_lens > time_step)[0], feature_sequence_lens
                output_mask[np.where(feature_sequence_lens > time_step)[0]+start_index,:] = 1
#            update_gate[start_index:end_index] += np.dot(hiddens[prev_index:start_index], model.weights['prevhidden_updategate'])
#            forget_gate[start_index:end_index] += np.dot(cell[prev_index:start_index], model.weights['prevcell_forgetgate'])
#            expit(update_gate[start_index:end_index], update_gate[start_index:end_index])
#            update_gate[start_index:end_index] *= mask
            
#            activation[start_index:end_index] += (np.dot(hiddens[prev_index:start_index] * reset_gate[start_index:end_index], 
#                                                         model.weights['prevhidden_activation']))
#            np.tanh(activation[start_index:end_index], activation[start_index:end_index])
#            activation[start_index:end_index] *= mask
#            cell[start_index:end_index] = part_cell[start_index:end_index] * input_gate[start_index:end_index] + forget_gate[start_index:end_index] * cell[prev_index:start_index]
#            part_cell[start_index:end_index] *= mask
#            cell[start_index:end_index] *= mask
#            
#            output_gate[start_index:end_index] += np.dot(hiddens[prev_index:start_index], model.weights['hidden_outputgate'])
#            output_gate[start_index:end_index] += np.dot(cell[start_index:end_index], model.weights['curcell_outputgate'])
#            expit(output_gate[start_index:end_index], output_gate[start_index:end_index])
#            output_gate[start_index:end_index] *= mask
            
#            hiddens[start_index:end_index] = (update_gate[start_index:end_index] * activation[start_index:end_index] +
#                                              (1-update_gate[start_index:end_index]) * hiddens[prev_index:start_index]) * mask
#            hiddens[start_index:end_index] *= mask
            
#        unordered_output = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']))
        if single_prediction:
            if not return_hiddens:
                indices = (feature_sequence_lens-1) * batch_size + range(feature_sequence_lens.size)
#                print indices
#                print hiddens.shape
#                print feature_sequence_lens
#                print batch_size
                ordered_hiddens = hiddens[(indices)]
                output = self.softmax(self.weight_matrix_multiply(ordered_hiddens, model.weights['hidden_output'], model.bias['output']))
            else:
                unordered_output = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']))
                output_mask = np.zeros(unordered_output.shape)
                indices = (feature_sequence_lens-1) * batch_size + range(feature_sequence_lens.size)
                output_mask[indices, :] = 1
#            if num_observations > 1:
#                output = self.softmax(self.weight_matrix_multiply(hiddens[-1:], model.weights['hidden_output'], model.bias['output']))
#            else:
#                output = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']))
        elif return_hiddens:
            unordered_output = self.softmax(self.weight_matrix_multiply(hiddens, model.weights['hidden_output'], model.bias['output']))
                
        if return_hiddens: #return unordered output for backprop
#            mask = np.zeros(unordered_output.shape)
#            if not single_prediction:
#                raise ValueError('multi-prediction not yet implemented')
#            indices = (feature_sequence_lens-1) * batch_size + range(feature_sequence_lens.size)
#            mask[indices, :] = 1
#            print mask
            
            return unordered_output * output_mask, hiddens
        else:
            del hiddens
            return output
        
    def forward_pass(self, inputs, feature_sequence_lens, model=None, return_hiddens=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
#        max_sequence_observations = max(feature_sequence_lens)
        num_sequences = len(feature_sequence_lens)
#        num_hiddens = architecture[1]
        num_outs = architecture[-1]
        if return_hiddens:
            print "returning hiddens for multiple sequences at a time is not yet implemented... try forward_pass_single_batch"
            raise ValueError
#            hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((sum(feature_sequence_lens), num_outs))
        #propagate hiddens
        cur_index = 0
        end_index = cur_index
        for sequence_index, feature_sequence_len in enumerate(feature_sequence_lens):
#            print sequence_index, feature_sequence_len
            end_index += feature_sequence_len
            if self.dssm:
                batch_inputs = inputs[cur_index:end_index]
            else:
                batch_inputs = inputs[:feature_sequence_len, sequence_index]
            outputs[cur_index:end_index, :] = self.forward_pass_single_batch(batch_inputs, model, return_hiddens)
#            else:
#                outputs[:feature_sequence_len, sequence_index, :], hiddens[:feature_sequence_len, sequence_index, :] = self.forward_pass_single_batch(inputs[cur_index:end_index], model, 
#                                                                                                                                                      return_hiddens)
            cur_index = end_index

        if return_hiddens:
            return outputs, hiddens
        else:
#            del hiddens
            return outputs
    
    def forward_pass_multi_output(self, inputs, feature_sequence_lens, model=None, return_hiddens=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        if return_hiddens:
            print "returning hiddens for multiple sequences at a time is not yet implemented... try forward_pass_single_batch"
            raise ValueError
#            hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        architecture = self.model.get_architecture()
        max_sequence_observations = max(feature_sequence_lens)
        num_sequences = len(feature_sequence_lens)
        num_hiddens = architecture[2]
        num_outs = architecture[3]
        if return_hiddens:
            hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        #propagate hiddens
        cur_index = 0
        end_index = cur_index
        for sequence_index, feature_sequence_len in enumerate(feature_sequence_lens):
            end_index += feature_sequence_len
#            if not return_hiddens:
            if self.dssm:
                batch_inputs = inputs[cur_index:end_index]
            else:
                batch_inputs = inputs[:feature_sequence_len, sequence_index]
            outputs[:feature_sequence_len, sequence_index, :] = self.forward_pass_single_batch(batch_inputs, model, return_hiddens, single_prediction = True)
#            else:
#                outputs[:feature_sequence_len, sequence_index, :], hiddens[:feature_sequence_len, sequence_index, :] = self.forward_pass_single_batch(inputs[cur_index:end_index], model, 
#                                                                                                                                                      return_hiddens, single_prediction = False)
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

    def calculate_cross_entropy(self, output, labels): #completed, expensive, should be compiled
        """calculates perplexity with flat labels
        """
        return -np.sum(np.log([max(output.item((x,labels[x])),1E-12) for x in range(labels.size)]))

    def calculate_cross_entropy_multi_output(self, output, feature_sequence_lengths, labels): #completed, expensive, should be compiled
        """calculates perplexity with flat labels
        """
        return -np.sum([np.sum(np.log(np.clip(output[:fsl, idx, labels[idx]], a_min=1E-12, a_max=1.0))) for idx,fsl in enumerate(feature_sequence_lengths)])

    def calculate_classification_accuracy(self, flat_output, labels): #completed, possibly expensive
        prediction = flat_output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = sum(prediction == labels) / float(labels.size)
        return classification_accuracy[0]
    

class RNNLM_Neural_Network_Addressee_Tester(RNNLM_Neural_Network_Addressee): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(RNNLM_Neural_Network_Addressee_Tester,self).__init__(config_dictionary)
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
#        if self.single_prediction:
#            self.classify()
#        else:
#            self.classify_multi_output()
        self.write_posterior_prob_file()
#        self.classify_log_perplexity()
#        self.write_log_perplexity_file()
    
    def classify(self): #completed
        self.posterior_probs = self.forward_pass_multi_batch(self.features, self.feature_sequence_lens, single_prediction = self.single_prediction)
        try:
            avg_cross_entropy = self.calculate_cross_entropy(self.flat_posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is %f%%" % self.calculate_classification_accuracy(self.flat_posterior_probs, self.labels) * 100
        except AttributeError:
            print "no labels given, so skipping classification statistics"
    
    def classify_multi_output(self): #completed
        self.posterior_probs = self.forward_pass_multi_output(self.features, self.feature_sequence_lens)
#        print self.posterior_probs.shape
        self.flat_posterior_probs = self.flatten_output(self.posterior_probs, self.feature_sequence_lens)
#        print self.flat_posterior_probs.shape
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

class RNNLM_Neural_Network_Addressee_Trainer(RNNLM_Neural_Network_Addressee):
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
        super(RNNLM_Neural_Network_Addressee_Trainer,self).__init__(config_dictionary)
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
            self.validation_features, self.validation_fsl, self.validation_dssm = self.read_feature_file(self.validation_feature_file_name)
            assert self.dssm == self.validation_dssm
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
            
            self.num_hiddens = self.default_variable_define(config_dictionary, 'num_hiddens', arg_type='int_comma_string', exit_if_no_default=True)
            if self.dssm:
                architecture = [self.features.shape[1]] + self.num_hiddens #+1 because index starts at 0
            else:
                architecture = [np.max(self.features)+1] + self.num_hiddens
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels[:,1])+1) #will have to change later if I have soft weights
#                architecture.append(np.max(self.labels.ravel())+1) #will have to change later if I have soft weights
#            print architecture
            self.seed = self.default_variable_define(config_dictionary, 'seed', 'int', '0')
#            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
#            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
#            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=0.1)
#            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-0.1)
            
            self.model.init_random_weights(architecture, seed = self.seed) #, self.initial_bias_max, self.initial_bias_min, 
                                           #self.initial_weight_min, self.initial_weight_max, maxent=self.use_maxent)
            del architecture #we have it in the model
        #
        self.use_maxent = self.default_variable_define(config_dictionary, 'use_maxent', arg_type='boolean', default_value=False)
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
    
    def calculate_gradient_single_batch(self, batch_inputs, batch_label, gradient_weights, dropout = 0.0, 
                                        check_gradient = False, model = None, l2_regularization_const = 0.0, 
                                        return_cross_entropy = False, single_prediction = True, clear_gradient = True): 
        #need to check regularization
        #TO DO: fix gradient when there is only a single word (empty?)
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        if dropout < 0.0 or dropout > 1.0:
            print "dropout must be between 0 and 1 but is", dropout
            raise ValueError
        
        if model == None:
            model = self.model
        output, embedding, input_gate, forget_gate, output_gate, part_cell, cell, hiddens = self.forward_pass_single_batch(batch_inputs, model, 
                                                                                                                           return_hiddens=True,
                                                                                                                           single_prediction = single_prediction)
        if dropout != 0.0:
            hiddens *= (np.random.rand(hiddens.shape[0], hiddens.shape[1]) > dropout)
#        print output, batch_label
        if return_cross_entropy:
            cross_entropy = -np.log2(output[0][batch_label])
        #derivative of log(cross-entropy softmax)
        num_observations = batch_inputs.shape[0]
        if clear_gradient:
            gradient_weights *= 0.0
        delta_output = output
        if self.single_prediction:
            delta_output[0][batch_label] -= 1.0
        else:
            for x in range(delta_output.shape[0]):
                delta_output[x][batch_label] -= 1.0
        
        eps_hidden = np.empty((batch_inputs.shape[0], model.get_architecture()[2]))
        eps_state = np.empty(eps_hidden.shape)
        delta_output_gate = np.empty(eps_hidden.shape)
        delta_input_gate = np.empty(eps_hidden.shape)
        delta_forget_gate = np.empty(eps_hidden.shape)
        delta_cell = np.empty(eps_hidden.shape)
        
#        if num_observations > 1:
        step = num_observations-1
#        print 1 / (1-input_gate[step])
#        quit()
        part_cell_deriv = (1 + part_cell[step]) * (1 - part_cell[step])
        cell_deriv = 1 - np.tanh(cell[step]) ** 2
        if self.single_prediction:
            eps_hidden[step] = np.dot(delta_output, model.weights['hidden_output'].T)
        else:
            eps_hidden[step] = np.dot(delta_output[step], model.weights['hidden_output'].T)
        delta_output_gate[step] = eps_hidden[step] * (1 - output_gate[step]) * hiddens[step]
        eps_state[step] = (eps_hidden[step] * output_gate[step] * cell_deriv
                           + np.dot(delta_output_gate[step], model.weights['curcell_outputgate'].T))
        delta_cell[step] = eps_state[step] * input_gate[step] * part_cell_deriv
        if num_observations > 1:
            delta_forget_gate[step] = eps_state[step] * cell[step-1] * forget_gate[step] * (1 - forget_gate[step])
        else:
            delta_forget_gate[0] = 0.0
        delta_input_gate[step] = eps_state[step] * part_cell[step] * input_gate[step] * (1 - input_gate[step])
        
        for step in range(num_observations-2, -1, -1):
            part_cell_deriv = (1 + part_cell[step]) * (1 - part_cell[step])
            cell_deriv = 1 - np.tanh(cell[step]) ** 2
            eps_hidden[step] = (np.dot(delta_output_gate[step+1], model.weights['hidden_outputgate'].T)
                                + np.dot(delta_input_gate[step+1], model.weights['hidden_inputgate'].T)
                                + np.dot(delta_forget_gate[step+1], model.weights['hidden_forgetgate'].T)
                                + np.dot(delta_cell[step+1], model.weights['hidden_cell'].T))
            if not self.single_prediction:
                eps_hidden[step] += np.dot(delta_output[step], model.weights['hidden_output'].T)
            delta_output_gate[step] = eps_hidden[step] * (1 - output_gate[step]) * hiddens[step]
            eps_state[step] = (eps_hidden[step] * output_gate[step] * cell_deriv
                               + eps_state[step+1] * forget_gate[step+1]
                               + np.dot(delta_output_gate[step], model.weights['curcell_outputgate'].T)
                               + np.dot(delta_forget_gate[step+1], model.weights['prevcell_forgetgate'].T)
                               + np.dot(delta_input_gate[step+1], model.weights['prevcell_inputgate'].T))
            delta_cell[step] = eps_state[step] * input_gate[step] * part_cell_deriv
            if step != 0:
                delta_forget_gate[step] = eps_state[step] * cell[step-1] * forget_gate[step] * (1 - forget_gate[step])
            else:
                delta_forget_gate[0] = 0.0
            delta_input_gate[step] = eps_state[step] * part_cell[step] * input_gate[step] * (1 - input_gate[step])
        #
        delta_embedding = np.dot(delta_output_gate, model.weights['visible_outputgate'].T)
        delta_embedding += np.dot(delta_input_gate, model.weights['visible_inputgate'].T)
        delta_embedding += np.dot(delta_forget_gate, model.weights['visible_forgetgate'].T)
        delta_embedding += np.dot(delta_cell, model.weights['visible_cell'].T)
#        self.bias_keys = ['inputgate', 'forgetgate', 'outputgate', 'cell', 'output']
        gradient_weights.bias['visible_hidden'] += np.sum(delta_embedding, axis = 0)
        gradient_weights.bias['inputgate'] += np.sum(delta_input_gate, axis = 0)
        gradient_weights.bias['forgetgate'] += np.sum(delta_forget_gate, axis = 0)
        gradient_weights.bias['outputgate'] += np.sum(delta_output_gate, axis = 0)
        gradient_weights.bias['cell'] += np.sum(delta_cell, axis = 0)
        if self.single_prediction:
            gradient_weights.bias['output'] += delta_output
        else:
            gradient_weights.bias['output'] += np.sum(delta_output, axis = 0)
        
#        self.weights_keys = ['visible_inputgate', 'visible_forgetgate', 'visible_outputgate', 'visible_cell',
#                             'hidden_inputgate', 'hidden_inputgate', 'hidden_outputgate', 'hidden_cell',
#                             'prevcell_inputgate', 'prevcell_forgetgate', 'curcell_outputgate',
#                             'hidden_output']
        gradient_weights.weights['visible_inputgate'] += embedding.T.dot(delta_input_gate)
        gradient_weights.weights['visible_forgetgate'] += embedding[1:].T.dot(delta_forget_gate[1:])
        gradient_weights.weights['visible_outputgate'] += embedding.T.dot(delta_output_gate)
        gradient_weights.weights['visible_cell'] = embedding.T.dot(delta_cell)
        
        if not self.dssm:
            for step in range(num_observations):
                gradient_weights.weights['visible_hidden'][batch_inputs[step]] += delta_embedding[step]
            
        else: #DSSM
#            print batch_inputs.shape
#            print delta_embedding.shape
#            print batch_inputs.T.dot(delta_embedding).shape
#            print gradient_weights.weights['visible_hidden'].shape
            gradient_weights.weights['visible_hidden'] += batch_inputs.T.dot(delta_embedding)
        
        if num_observations > 1:
            gradient_weights.weights['hidden_cell'] += np.dot(hiddens[:-1].T, delta_cell[1:])
            gradient_weights.weights['hidden_inputgate'] += np.dot(hiddens[:-1].T, delta_input_gate[1:])
            gradient_weights.weights['hidden_forgetgate'] += np.dot(hiddens[:-1].T, delta_forget_gate[1:])
            gradient_weights.weights['hidden_outputgate'] += np.dot(hiddens[:-1].T, delta_output_gate[1:])
            gradient_weights.weights['prevcell_inputgate'] += np.dot(cell[:-1].T, delta_input_gate[1:])
            gradient_weights.weights['prevcell_forgetgate'] += np.dot(cell[:-1].T, delta_forget_gate[1:])
#        else: #not needed because gradient is already set to 0.0
#            gradient_weights.weights['hidden_cell'] = 0.0
#            gradient_weights.weights['hidden_inputgate'] = 0.0
#            gradient_weights.weights['hidden_forgetgate'] = 0.0
#            gradient_weights.weights['hidden_outputgate'] = 0.0
#            gradient_weights.weights['prevcell_inputgate'] = 0.0
#            gradient_weights.weights['prevcell_forgetgate'] = 0.0
        
        
        gradient_weights.weights['curcell_outputgate'] += np.dot(cell.T, delta_output_gate)
        
        
        if self.single_prediction:
            gradient_weights.weights['hidden_output'] += np.outer(hiddens[-1], output)
            delta_output[0][batch_label] += 1.0
        else:
            gradient_weights.weights['hidden_output'] += np.dot(hiddens.T, output)
            for x in range(delta_output.shape[0]):
                delta_output[x][batch_label] += 1.0
#        print batch_inputs
#        print batch_labels
#        print batch_indices
        
        
        if not check_gradient:
            if not self.single_prediction:
                gradient_weights /= num_observations
            if not return_cross_entropy:
                if l2_regularization_const > 0.0:
                    gradient_weights += model * l2_regularization_const
                return
            else:
                if l2_regularization_const > 0.0:
                    return gradient_weights + model * l2_regularization_const, cross_entropy
                return cross_entropy
            
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * l2_regularization_const
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = RNNLM_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
#            print "at initial hiddens"
#            for index in range(direction.init_hiddens.size):
#                direction.init_hiddens[0][index] = epsilon
#                forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
#                backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
#                finite_difference_model.init_hiddens[0][index] = (forward_loss - backward_loss) / (2 * epsilon)
#                direction.init_hiddens[0][index] = 0.0
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
                    forward_loss = -np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[0][batch_label])
                    backward_loss = -np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[0][batch_label])
                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        forward_loss = -np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[0][batch_label])
                        backward_loss = -np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[0][batch_label])
                        finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                        direction.weights[key][index0][index1] = 0.0

            for bias_name in self.model.bias_keys:
                print "calculated gradient for %s bias" % bias_name
                print gradient_weights.bias[bias_name]
                print "finite difference approximation for %s bias" % bias_name
                print finite_difference_model.bias[bias_name]
                
            for weight_name in self.model.weights_keys:
                print "calculated gradient for %s weights" % weight_name
                print gradient_weights.weights[weight_name]
                print "finite difference approximation for %s weights" % weight_name
                print finite_difference_model.weights[weight_name]
            
            for bias_name in self.model.bias_keys:
                print "gradient / finite_difference for %s bias" % bias_name
                print gradient_weights.bias[bias_name] / (finite_difference_model.bias[bias_name] + 1E-20)
            
            for weight_name in self.model.weights_keys:
                print "gradient / finite_difference for %s weights" % weight_name
                print gradient_weights.weights[weight_name] / (finite_difference_model.weights[weight_name] + 1E-20)
                
            sys.exit()
        ##########################################################
    
    def calculate_gradient_multi_batch(self, batch_inputs, fsl, batch_labels, gradient_weights, dropout = 0.0, 
                                        check_gradient = False, model = None, l2_regularization_const = 0.0, 
                                        return_cross_entropy = False, single_prediction = True, clear_gradient = True): 
        #need to check regularization
        #TO DO: fix gradient when there is only a single word (empty?)
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        if dropout < 0.0 or dropout > 1.0:
            print "dropout must be between 0 and 1 but is", dropout
            raise ValueError
        
        if model == None:
            model = self.model
        
        num_observations, batch_size = max(fsl), fsl.size
        
        
        output, hiddens = self.forward_pass_multi_batch(batch_inputs, fsl, model, return_hiddens = True,
                                                        single_prediction = single_prediction)
        
#        output_multi = self.forward_pass_multi_batch(batch_inputs, fsl, model, return_hiddens = False,
#                                                     single_prediction = single_prediction)
#        
#        output_ref = self.forward_pass(batch_inputs, fsl, model, return_hiddens=False)
#        print "***************************"
#        print fsl
#        print output_multi - output_ref
#        print "***************************"
        
        indices = (fsl - 1) * batch_size + range(fsl.size)
        
        if dropout != 0.0:
            hiddens *= (np.random.rand(hiddens.shape[0], hiddens.shape[1]) > dropout)
#        print output, batch_label
        if return_cross_entropy:
            cross_entropy = np.sum(-np.log2(output[indices, batch_labels]))
        #derivative of log(cross-entropy softmax)
        
        if clear_gradient:
            gradient_weights *= 0.0
        delta_output = output
        if self.single_prediction:
            delta_output[indices, batch_labels] -= 1.0
        else:
            frame_labels = np.tile(batch_labels, (num_observations,))
            output_mask = np.where(np.sum(output, axis = 1) == 0.0)[0]
#            print np.sum(output, axis = 1)
#            print output_mask
            delta_output[np.arange(frame_labels.size), frame_labels] -= 1.0
            delta_output[(output_mask), :] = 0
#            raise ValueError('multiple outputs not yet implemented')
#            for x in range(delta_output.shape[0]):
#                delta_output[x][batch_labels] -= 1.0
        mask = np.ones((batch_size, model.bias['hidden'].size))
#        ones_vec = np.ones((model.get_architecture()[2],))
        eps_hidden = np.empty((num_observations * batch_size, model.bias['hidden'].size))
#        delta_activation = np.empty(eps_hidden.shape)
#        delta_update_gate = np.empty(eps_hidden.shape)
#        delta_reset_gate = np.empty(eps_hidden.shape)
#        delta_forget_gate = np.empty(eps_hidden.shape)
#        delta_cell = np.empty(eps_hidden.shape)
        
#        if num_observations > 1:
        step = num_observations-1
        mask[(fsl <= step), :] = 0.0
        start_index = batch_size * step
        end_index = batch_size * (step + 1)
#        prev_index = batch_size * (step - 1)
        hidden_deriv = hiddens[start_index:end_index] * (1 - hiddens[start_index:end_index]) * mask
#        print 1 / (1-input_gate[step])
#        quit()
#        activation_deriv = (1 + activation[start_index:end_index]) * (1 - activation[start_index:end_index]) * mask
#        update_gate_deriv = update_gate[start_index:end_index] * (1 - update_gate[start_index:end_index]) * mask
#        reset_gate_deriv = reset_gate[start_index:end_index] * (1 - reset_gate[start_index:end_index]) * mask
#        cell_deriv = 1 - np.tanh(cell[start_index:end_index]) ** 2
#        activation_deriv *= mask
#        cell_deriv *= mask
        
        np.dot(delta_output, model.weights['hidden_output'].T, out = eps_hidden)
        
        eps_hidden[start_index:end_index] *= hidden_deriv * mask
#        delta_update_gate[start_index:end_index] = (eps_hidden[start_index:end_index] * (activation[start_index:end_index] - hiddens[prev_index:start_index])
#                                                    * update_gate_deriv)
##        delta_update_gate[start_index:end_index] *= mask
#        
#        delta_activation[start_index:end_index] = (eps_hidden[start_index:end_index] * update_gate[start_index:end_index] *
#                                                   activation_deriv * mask)
#        if num_observations > 1:
#            delta_reset_gate[start_index:end_index] = (np.dot(delta_activation[start_index:end_index], 
#                                                              model.weights['prevhidden_activation'].T) * 
#                                                       hiddens[prev_index:start_index] * 
#                                                       reset_gate_deriv) * mask
#        else:
#            delta_reset_gate[start_index:end_index] = 0.0
        
#        eps_state[start_index:end_index] = (eps_hidden[start_index:end_index] * output_gate[start_index:end_index] * cell_deriv
#                                            + np.dot(delta_output_gate[start_index:end_index], model.weights['curcell_outputgate'].T))
#        eps_state[start_index:end_index] *= mask
#        delta_cell[start_index:end_index] = eps_state[start_index:end_index] * input_gate[start_index:end_index] * part_cell_deriv
#        delta_cell[start_index:end_index] *= mask
#        if num_observations > 1:
#            delta_forget_gate[start_index:end_index] = (eps_state[start_index:end_index] * cell[prev_index:start_index] * 
#                                                        forget_gate[start_index:end_index] * (1 - forget_gate[start_index:end_index]))
#        else:
#            delta_forget_gate[start_index:end_index] = 0.0
#        delta_input_gate[start_index:end_index] = (eps_state[start_index:end_index] * part_cell[start_index:end_index] * 
#                                                   input_gate[start_index:end_index] * (1 - input_gate[start_index:end_index]))
#        delta_input_gate[start_index:end_index] *= mask
#        delta_forget_gate[start_index:end_index] *= mask
        
        for step in range(num_observations-2, -1, -1):
            mask[:] = 1.0
            mask[(fsl <= step), :] = 0.0
            start_index = batch_size * step
            end_index = batch_size * (step + 1)
#            prev_index = batch_size * (step - 1)
            next_index = batch_size * (step + 2)
            
            hidden_deriv = hiddens[start_index:end_index] * (1 - hiddens[start_index:end_index]) * mask
#            update_gate_deriv = update_gate[start_index:end_index] * (1 - update_gate[start_index:end_index]) * mask
#            reset_gate_deriv = reset_gate[start_index:end_index] * (1 - reset_gate[start_index:end_index]) * mask
            
#            part_cell_deriv = (1 + part_cell[start_index:end_index]) * (1 - part_cell[start_index:end_index])
#            cell_deriv = 1 - np.tanh(cell[start_index:end_index]) ** 2
#            part_cell_deriv *= mask
#            cell_deriv *= mask
            eps_hidden[start_index:end_index] += np.dot(eps_hidden[end_index:next_index], model.weights['hidden_hidden'].T)
#                                                 + np.dot(delta_output[start_index:end_index], model.weights['hidden_output'].T))
            eps_hidden[start_index:end_index] *= hidden_deriv
#            if step > 0:
#                delta_update_gate[start_index:end_index] = (eps_hidden[start_index:end_index] * (activation[start_index:end_index] - hiddens[prev_index:start_index])
#                                                            * update_gate_deriv * mask)
#            else:
#                delta_update_gate[start_index:end_index] = (eps_hidden[start_index:end_index] * activation[start_index:end_index]
#                                                            * update_gate_deriv * mask)
    #        delta_update_gate[start_index:end_index] *= mask
            
#            delta_activation[start_index:end_index] = (eps_hidden[start_index:end_index] * update_gate[start_index:end_index] *
#                                                       activation_deriv * mask)
#            if step != 0:
#                delta_reset_gate[start_index:end_index] = (np.dot(delta_activation[start_index:end_index], 
#                                                                  model.weights['prevhidden_activation'].T) * 
#                                                           hiddens[prev_index:start_index] * 
#                                                           reset_gate_deriv * mask)
#            else:
#                delta_reset_gate[start_index:end_index] = 0.0
#            if not self.single_prediction:
#                eps_hidden[start_index:end_index] += np.dot(delta_output[start_index:end_index], model.weights['hidden_output'].T)
            
#            delta_output_gate[start_index:end_index] = eps_hidden[start_index:end_index] * (1 - output_gate[start_index:end_index]) * hiddens[start_index:end_index]
#            delta_output_gate[start_index:end_index] *= mask
#            eps_state[start_index:end_index] = (eps_hidden[start_index:end_index] * output_gate[start_index:end_index] * cell_deriv
#                                                + eps_state[end_index:next_index] * forget_gate[end_index:next_index]
#                                                + np.dot(delta_output_gate[start_index:end_index], model.weights['curcell_outputgate'].T)
#                                                + np.dot(delta_forget_gate[end_index:next_index], model.weights['prevcell_forgetgate'].T)
#                                                + np.dot(delta_input_gate[end_index:next_index], model.weights['prevcell_inputgate'].T))
#            eps_state[start_index:end_index] *= mask
#            delta_cell[start_index:end_index] = eps_state[start_index:end_index] * input_gate[start_index:end_index] * part_cell_deriv
#            delta_cell[start_index:end_index] *= mask
#            if step != 0:
#                delta_forget_gate[start_index:end_index] = (eps_state[start_index:end_index] * cell[prev_index:start_index] * 
#                                                            forget_gate[start_index:end_index] * (1 - forget_gate[start_index:end_index]))
#            else:
#                delta_forget_gate[start_index:end_index] = 0.0
#            delta_forget_gate[start_index:end_index] *= mask
#            delta_input_gate[start_index:end_index] = (eps_state[start_index:end_index] * part_cell[start_index:end_index] * 
#                                                       input_gate[start_index:end_index] * (1 - input_gate[start_index:end_index]))
#            delta_input_gate[start_index:end_index] *= mask
        #
#        print eps_hidden
#        print batch_inputs.ravel()
#        print np.sum(eps_hidden, axis = 0)
        if not self.dssm:
            out_mat = ssp.dok_matrix((model.weights['visible_hidden'].shape[0], batch_inputs.size))
            out_mat[(batch_inputs.ravel()), (np.arange(batch_inputs.size))] = 1.0
            gradient_weights.weights['visible_hidden'] = out_mat.dot(eps_hidden)
        else:
            gradient_weights.weights['visible_hidden'] += self.make_dssm_batch_features(batch_inputs, fsl).T.dot(eps_hidden)
        gradient_weights.bias['hidden'] += np.sum(eps_hidden, axis = 0)
#        delta_embedding = np.dot(delta_update_gate, model.weights['visible_updategate'].T)
#        delta_embedding += np.dot(delta_reset_gate, model.weights['visible_resetgate'].T)
#        delta_embedding += np.dot(delta_activation, model.weights['visible_activation'].T)
#        delta_embedding += np.dot(delta_cell, model.weights['visible_cell'].T)
#        self.bias_keys = ['inputgate', 'forgetgate', 'outputgate', 'cell', 'output']
#        gradient_weights.bias['visible_hidden'] += np.sum(delta_embedding, axis = 0)
#        gradient_weights.bias['updategate'] += np.sum(delta_update_gate, axis = 0)
#        gradient_weights.bias['resetgate'] += np.sum(delta_reset_gate, axis = 0)
#        gradient_weights.bias['outputgate'] += np.sum(delta_output_gate, axis = 0)
#        gradient_weights.bias['activation'] += np.sum(delta_activation, axis = 0)
        
        gradient_weights.bias['output'] += np.sum(delta_output, axis = 0)
        gradient_weights.init_hiddens += np.sum(np.dot(eps_hidden[:batch_size], model.weights['hidden_hidden'].T), 
                                                axis = 0)
        
#        if self.single_prediction:
#            gradient_weights.bias['output'] += delta_output
#        else:
#            gradient_weights.bias['output'] += np.sum(delta_output, axis = 0)
        
#        self.weights_keys = ['visible_inputgate', 'visible_forgetgate', 'visible_outputgate', 'visible_cell',
#                             'hidden_inputgate', 'hidden_inputgate', 'hidden_outputgate', 'hidden_cell',
#                             'prevcell_inputgate', 'prevcell_forgetgate', 'curcell_outputgate',
#                             'hidden_output']
#        gradient_weights.weights['visible_updategate'] += embedding.T.dot(delta_update_gate)
#        gradient_weights.weights['visible_forgetgate'] += embedding[batch_size:].T.dot(delta_forget_gate[batch_size:])
#        gradient_weights.weights['visible_resetgate'] += embedding.T.dot(delta_reset_gate)
#        gradient_weights.weights['visible_activation'] = embedding.T.dot(delta_activation)
        
#        if not self.dssm:
#            linear_inputs = batch_inputs.ravel()
#            for step in range(linear_inputs.size):
#                gradient_weights.weights['visible_hidden'][linear_inputs[step]] += delta_embedding[step]
#            
#        else: #DSSM
#            print batch_inputs.shape
#            print delta_embedding.shape
#            print batch_inputs.T.dot(delta_embedding).shape
#            print gradient_weights.weights['visible_hidden'].shape
#            raise ValueError('THIS SHOULD NOT DSSM')
#            gradient_weights.weights['visible_hidden'] += batch_inputs.T.dot(delta_embedding)
        
        if num_observations > 1:
            gradient_weights.weights['hidden_hidden'] += np.dot(hiddens[:-batch_size].T, eps_hidden[batch_size:])
            gradient_weights.weights['hidden_hidden'] += np.dot(np.tile(model.init_hiddens, (batch_size, 1)).T, 
                                                                eps_hidden[:batch_size])
#            gradient_weights.weights['prevhidden_activation'] += np.dot((hiddens[:-batch_size]*reset_gate[batch_size:]).T, delta_activation[batch_size:])
#            gradient_weights.weights['prevhidden_updategate'] += np.dot(hiddens[:-batch_size].T, delta_update_gate[batch_size:])
#            gradient_weights.weights['prevhidden_resetgate'] += np.dot(hiddens[:-batch_size].T, delta_reset_gate[batch_size:])
#            gradient_weights.weights['hidden_outputgate'] += np.dot(hiddens[:-batch_size].T, delta_output_gate[batch_size:])
#            gradient_weights.weights['prevcell_inputgate'] += np.dot(cell[:-batch_size].T, delta_input_gate[batch_size:])
#            gradient_weights.weights['prevcell_forgetgate'] += np.dot(cell[:-batch_size].T, delta_forget_gate[batch_size:])
#        else: #not needed because gradient is already set to 0.0
#            gradient_weights.weights['hidden_cell'] = 0.0
#            gradient_weights.weights['hidden_inputgate'] = 0.0
#            gradient_weights.weights['hidden_forgetgate'] = 0.0
#            gradient_weights.weights['hidden_outputgate'] = 0.0
#            gradient_weights.weights['prevcell_inputgate'] = 0.0
#            gradient_weights.weights['prevcell_forgetgate'] = 0.0
        
        
#        gradient_weights.weights['curcell_outputgate'] += np.dot(cell.T, delta_output_gate)
#        print hiddens
#        print fsl
#        print batch_labels
#        print delta_output
        gradient_weights.weights['hidden_output'] = np.dot(hiddens.T, delta_output)
        if self.single_prediction:
#            gradient_weights.weights['hidden_output'] += np.outer(hiddens[-1], output)
            delta_output[indices, batch_labels] += 1.0
            delta_output[(output_mask), :] = 0
#            delta_output[0][batch_label] += 1.0
        else:
            delta_output[np.arange(frame_labels.size), frame_labels] += 1.0
#            gradient_weights.weights['hidden_output'] += np.dot(hiddens.T, output)
#            for x in range(delta_output.shape[0]):
#                delta_output[x][batch_label] += 1.0
#        print batch_inputs
#        print batch_labels
#        print batch_indices
        
        
        if not check_gradient:
            if not self.single_prediction:
                gradient_weights /= fsl.sum()
            else:
                gradient_weights /= fsl.size
            if not return_cross_entropy:
                if l2_regularization_const > 0.0:
                    gradient_weights += model * l2_regularization_const
                return
            else:
                if l2_regularization_const > 0.0:
                    return gradient_weights + model * l2_regularization_const, cross_entropy
                return cross_entropy
            
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * l2_regularization_const
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = RNNLM_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
#            print "at initial hiddens"
#            for index in range(direction.init_hiddens.size):
#                direction.init_hiddens[0][index] = epsilon
#                forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
#                backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
#                finite_difference_model.init_hiddens[0][index] = (forward_loss - backward_loss) / (2 * epsilon)
#                direction.init_hiddens[0][index] = 0.0
#            indices = np.arange(batch_labels.size)
            frame_labels = self.make_frame_labels(fsl, batch_labels)
            num_frames = sum(fsl)
            indices = np.arange(num_frames)
#            print "INDICES, FRAME_LABELS"
#            print indices
#            print frame_labels
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
#                    print self.forward_pass_multi_batch(batch_inputs, fsl, model = model + direction).shape
#                    print indices
#                    print batch_labels
                    
                    forward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model + direction, single_prediction = False)[indices,frame_labels]))
                    backward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model - direction, single_prediction = False)[indices,frame_labels]))
                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        forward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model + direction, single_prediction = False)[indices,frame_labels]))
                        backward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model - direction, single_prediction = False)[indices,frame_labels]))
                        finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                        direction.weights[key][index0][index1] = 0.0
            
            for index1 in range(direction.init_hiddens.size):
                direction.init_hiddens[0][index1] = epsilon
                forward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model + direction, single_prediction = False)[indices,frame_labels]))
                backward_loss = -np.sum(np.log(self.forward_pass_multi_batch(batch_inputs, fsl, model = model - direction, single_prediction = False)[indices,frame_labels]))
                finite_difference_model.init_hiddens[0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                direction.init_hiddens[0][index1] = 0.0

            for bias_name in self.model.bias_keys:
                print "calculated gradient for %s bias" % bias_name
                print gradient_weights.bias[bias_name]
                print "finite difference approximation for %s bias" % bias_name
                print finite_difference_model.bias[bias_name]
                
            for weight_name in self.model.weights_keys:
                print "calculated gradient for %s weights" % weight_name
                print gradient_weights.weights[weight_name]
                print "finite difference approximation for %s weights" % weight_name
                print finite_difference_model.weights[weight_name]
            
            print "calculated gradient for init hiddens"
            print gradient_weights.init_hiddens
            print "finite difference approximation for init hiddens"
            print finite_difference_model.init_hiddens
            
            for bias_name in self.model.bias_keys:
                print "gradient / finite_difference for %s bias" % bias_name
                print gradient_weights.bias[bias_name] / (finite_difference_model.bias[bias_name] + 1E-20)
            
            for weight_name in self.model.weights_keys:
                print "gradient / finite_difference for %s weights" % weight_name
                print gradient_weights.weights[weight_name] / (finite_difference_model.weights[weight_name] + 1E-20)
            
            print "gradient / finite_difference for init hiddens"
            print gradient_weights.init_hiddens / (finite_difference_model.init_hiddens + 1E-20)
            sys.exit()
        ##########################################################
    
    
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
        num_correct = 0
        num_sequences = len(feature_sequence_lens)
        if self.single_prediction:
            num_examples = len(feature_sequence_lens)
        else:
            num_examples = sum(feature_sequence_lens)
#        print features.shape
        cumsum_fsl = np.concatenate((np.array([0]), np.cumsum(feature_sequence_lens.T)))
        start_frame = 0
        print "calculating classification statistics"
        while end_index < num_sequences: #run through the batches
            per_done = float(batch_index)/num_sequences*100
#            sys.stdout.write("\r                                                                \r") #clear line
#            sys.stdout.write("\rCalculating Classification Statistics: %.1f%% done " % per_done), sys.stdout.flush()
            end_index = min(batch_index+classification_batch_size, num_sequences)
#            max_seq_len = max(feature_sequence_lens[batch_index:end_index])
#            start_frame = np.where(flat_labels[:,0] == batch_index)[0][0]
            end_frame = cumsum_fsl[end_index]
            label = flat_labels[batch_index:end_index]
            end_index = min(batch_index+classification_batch_size, num_sequences)
            max_seq_len = max(feature_sequence_lens[batch_index:end_index])
#            print batch_index, max_seq_len
            if self.dssm:
#                print features
#                print start_frame, end_frame
#                print features[start_frame:end_frame].shape
#                print feature_sequence_lens[batch_index:end_index]
#                print feature_sequence_lens[:10]
#                print sum(feature_sequence_lens[batch_index:end_index])
                output = self.forward_pass_multi_batch(features[start_frame:end_frame], feature_sequence_lens[batch_index:end_index], model=model,
                                                       single_prediction = self.single_prediction)
#                if self.single_prediction:
#                    
#                else:
#                    output = self.forward_pass_multi_(features[start_frame:end_frame], feature_sequence_lens[batch_index:end_index], model=model)
            else:
                output = self.forward_pass_multi_batch(features[:max_seq_len,batch_index:end_index], 
                                                       feature_sequence_lens[batch_index:end_index], model=model,
                                                       single_prediction = self.single_prediction)
#                if self.single_prediction:
##                    output = self.forward_pass(features[:max_seq_len,batch_index:end_index], 
##                                               feature_sequence_lens[batch_index:end_index], model=model)
#                    output = self.forward_pass_multi_batch(features[:max_seq_len,batch_index:end_index], 
#                                                           feature_sequence_lens[batch_index:end_index], model=model)
#                else:
#                    output = self.forward_pass_multi_batch(features[:max_seq_len,batch_index:end_index], 
#                                                            feature_sequence_lens[batch_index:end_index], model=model,
#                                                            single_prediction = False)
            
            prediction = output.argmax(axis=1)
            if self.single_prediction:
                cross_entropy += self.calculate_cross_entropy(output, label[:,1])
                
                num_correct += np.sum(prediction == label[:,1])
            else:
#                print label
                frame_labels = self.make_frame_labels(feature_sequence_lens[batch_index:end_index], label[:,1])
#                print frame_labels
                cross_entropy += self.calculate_cross_entropy(output, frame_labels)
                num_correct += np.sum(prediction == frame_labels)
#                for seq, fsl in enumerate(feature_sequence_lens[batch_index:end_index]):
#                    prediction = output[:fsl,seq,:].argmax(axis=1)
#                    num_correct += np.sum(prediction == label[seq,1])
            #don't use calculate_classification_accuracy() because of possible rounding error
            #- (prediction.size - num_examples) #because of the way we handle features, where some observations are null, we want to remove those examples for calculating accuracy
            batch_index += classification_batch_size
            start_frame = end_frame
        
#        sys.stdout.write("\r                                                                \r") #clear line
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        
#        cross_entropy /= np.log(2) * num_examples
        loss /= np.log(2) * num_examples
        return cross_entropy, num_correct, num_examples, loss

    def backprop_steepest_descent_single_batch(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        self.dropout = 0.0
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        if self.validation_feature_file_name is not None:
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
        
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
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] *= 1. / (1 - self.dropout)
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                if self.dssm:
                    batch_features = self.features[start_frame:end_frame]
                else:
                    batch_features = self.features[:feature_sequence_len, batch_index]
                batch_label = self.labels[batch_index,1]
                
#                print batch_features
#                print batch_label
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_label, gradient, dropout = self.dropout,
                                                                return_cross_entropy = True, check_gradient = False, 
                                                                single_prediction = self.single_prediction)
#                print self.model.norm()
#                print gradient.norm()
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
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] /= 1. / (1 - self.dropout)
            if self.validation_feature_file_name is not None:
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
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
        
    def backprop_rmsprop_single_batch(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        self.dropout = 0.0
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        rms_weights = RNNLM_Weight()
        rms_weights.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        rms_weights = rms_weights + 1E-10
        if self.validation_feature_file_name is not None:
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
        
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
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] *= 1. / (1 - self.dropout)
            batch_size = 10
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                if self.dssm:
                    batch_features = self.features[start_frame:end_frame]
                else:
                    batch_features = self.features[:feature_sequence_len, batch_index]
                batch_label = self.labels[batch_index,1]
                #Nesterov instead of classical momentum
                if momentum_rate > 0.0:
                    prev_step *= momentum_rate
                    self.model += prev_step
#                print batch_features
#                print batch_label
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_label, gradient, dropout = self.dropout,
                                                                return_cross_entropy = True, check_gradient = False, 
                                                                single_prediction = self.single_prediction, clear_gradient = False)
#                print self.model.norm()
#                print gradient.norm()
                cross_entropy += cur_xent 
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                ppp = cross_entropy / end_frame
                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                if batch_index % batch_size != 0 and batch_index != len(self.feature_sequence_lens) - 1:
                    continue
                rms_weights = (gradient ** 2) * 0.1 + rms_weights * 0.9
                gradient *= -self.steepest_learning_rate[epoch_num]
                gradient /= (rms_weights ** 0.5)
                if self.l2_regularization_const > 0.0:
                    self.model *= (1-self.l2_regularization_const) #l2 regularization_const
                self.model += gradient #/ batch_size
                gradient *= 0.0
                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num]
                
                start_frame = end_frame
            print "Training for epoch finished at", datetime.datetime.now()
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] /= 1. / (1 - self.dropout)
            if self.validation_feature_file_name is not None:
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))           
            print "Epoch finished at", datetime.datetime.now()
        end_time = datetime.datetime.now()
        print "Training finished at", end_time, "and ran for", end_time - start_time
#    def unflatten_labels(self, labels, sentence_ids):
#        num_frames_per_sentence = np.bincount(sentence_ids)
#        num_outs = len(np.unique(labels))
#        max_num_frames_per_sentence = np.max(num_frames_per_sentence)
#        unflattened_labels = np.zeros((max_num_frames_per_sentence, np.max(sentence_ids) + 1, num_outs)) #add one because first sentence starts at 0
#        current_sentence_id = 0
#        observation_index = 0
#        for label, sentence_id in zip(labels,sentence_ids):
#            if sentence_id != current_sentence_id:
#                current_sentence_id = sentence_id
#                observation_index = 0
#            unflattened_labels[observation_index, sentence_id, label] = 1.0
#            observation_index += 1
#        return unflattened_labels


    def backprop_steepest_descent_single_batch_semi_newbob(self):
        print "Starting backprop using semi-newbob steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        self.dropout = 0.0
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
#        print self.model.get_architecture()
        if self.validation_feature_file_name is not None:
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
        learning_rate = self.steepest_learning_rate[0]
        if hasattr(self, 'momentum_rate'):
            momentum_rate = self.momentum_rate[0]
            print "momentum is", momentum_rate
        else:
            momentum_rate = 0.0
        num_decreases = 0
        prev_cross_entropy = cross_entropy
        prev_num_correct = num_correct
        is_init = True
        init_decreases = 0
        self.model.write_weights(''.join([self.output_name, '_best_weights']))
        for epoch_num in range(1000):
            print "At epoch", epoch_num+1, "with learning rate", learning_rate, "and momentum", momentum_rate
            print "Training for epoch started at", datetime.datetime.now()
            start_frame = 0
            end_frame = 0
            cross_entropy = 0.0
            num_examples = 0
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] *= 1. / (1 - self.dropout)
                
            for batch_index, feature_sequence_len in enumerate(self.feature_sequence_lens):
                end_frame = start_frame + feature_sequence_len
                if self.dssm:
                    batch_features = self.features[start_frame:end_frame]
                else:
                    batch_features = self.features[:feature_sequence_len, batch_index]
                batch_label = self.labels[batch_index,1]
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                
#                print batch_features
#                print batch_label
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_single_batch(batch_features, batch_label, gradient, dropout = self.dropout,
                                                                return_cross_entropy = True, check_gradient = False, 
                                                                single_prediction = self.single_prediction)
#                print self.model.norm()
#                print gradient.norm()
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
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] /= 1. / (1 - self.dropout)
            if self.validation_feature_file_name is not None:
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
            else:
                raise ValueError("validation feature file must exist")
#            print prev_cross_entropy, cross_entropy
            if cross_entropy + 0.001 * self.validation_features.shape[0] < prev_cross_entropy:
                is_init = False
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
                if is_init: init_decreases += 1
                if init_decreases == 15:
                    print "Tried to find initial learning rate, but failed, quitting"
                    break
                if not is_init: num_decreases += 1 #don't count num_decreases when trying to find initial learning rate
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

    def backprop_steepest_descent_multi_batch_semi_newbob(self):
        print "Starting backprop using semi-newbob steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        self.dropout = 0.0
        prev_step = RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        gradient = RNNLM_Weight()
        gradient.init_zero_weights(self.model.get_architecture(), maxent = self.use_maxent)
        print self.model.get_architecture()
        if self.validation_feature_file_name is not None:
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
            print "cross-entropy before steepest descent is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
        learning_rate = self.steepest_learning_rate[0]
        if hasattr(self, 'momentum_rate'):
            momentum_rate = self.momentum_rate[0]
            print "momentum is", momentum_rate
        else:
            momentum_rate = 0.0
        num_decreases = 0
        prev_cross_entropy = cross_entropy
        prev_num_correct = num_correct
        is_init = True
        init_decreases = 0
        self.model.write_weights(''.join([self.output_name, '_best_weights']))
        cumsum_fsl = np.cumsum(np.hstack((0, self.feature_sequence_lens)))
        for epoch_num in range(1000):
            print "At epoch", epoch_num+1, "with learning rate", learning_rate, "and momentum", momentum_rate
            print "Training for epoch started at", datetime.datetime.now()
#            start_frame = 0
#            end_frame = 0
            cross_entropy = 0.0
            batch_index = 0
            end_index = 0
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] *= 1. / (1 - self.dropout)
                
            while end_index < self.num_sequences:
                
                end_index = min(batch_index+self.backprop_batch_size, self.num_sequences)
                batch_fsl = self.feature_sequence_lens[batch_index:end_index]
#                max_fsl = max(batch_fsl)
#                batch_features = self.features[:max_fsl, batch_index:end_index]
                batch_labels = self.labels[batch_index:end_index,1]
                
                if type(self.features) == ssp.csr_matrix:
                    start_frame = cumsum_fsl[batch_index]
                    end_frame = cumsum_fsl[end_index]
                    batch_features = self.features[start_frame:end_frame]
                else:
                    max_fsl = max(batch_fsl)
                    batch_features = self.features[:max_fsl, batch_index:end_index]
                
#                print batch_features
#                print batch_label
#                print ""
#                print batch_index
#                print batch_features
#                print batch_labels
                cur_xent = self.calculate_gradient_multi_batch(batch_features, batch_fsl, batch_labels, gradient, 
                                                               dropout = self.dropout,
                                                               return_cross_entropy = True, check_gradient = False, 
                                                               single_prediction = self.single_prediction)
#                print gradient
#                quit()
#                print self.model.norm()
#                print gradient.norm()
                cross_entropy += cur_xent 
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                ppp = cross_entropy / end_index
                sys.stdout.write("train X-ent: %f " % ppp), sys.stdout.flush()
                gradient *= -learning_rate
                if self.l2_regularization_const > 0.0:
                    self.model *= (1-self.l2_regularization_const) #l2 regularization_const
                self.model += gradient #/ batch_size
                if momentum_rate > 0.0:
                    prev_step *= momentum_rate
                    self.model += prev_step
                prev_step.assign_weights(gradient)
#                prev_step *= -self.steepest_learning_rate[epoch_num]
                batch_index = end_index
#                start_frame = end_frame
            print "Training for epoch finished at", datetime.datetime.now()
            if self.dropout != 0.0:
                self.model.weights['hidden_output'] /= 1. / (1 - self.dropout)
            if self.validation_feature_file_name is not None:
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.validation_features, self.validation_labels, self.validation_fsl, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is %d of %d (%.2f%%)" % (num_correct, num_examples, 100.0 * num_correct / num_examples)
            else:
                raise ValueError("validation feature file must exist")
#            print prev_cross_entropy, cross_entropy
            if cross_entropy * 1.001 < prev_cross_entropy: #cross_entropy + 0.001 * self.validation_features.shape[0] < prev_cross_entropy:
                is_init = False
                prev_cross_entropy = cross_entropy
                prev_num_correct = num_correct
                self.model.write_weights(''.join([self.output_name, '_best_weights']))
                if self.save_each_epoch:
                    self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
                print "num_decreases so far is", num_decreases
                if num_decreases == 5: 
                    learning_rate /= 2.0
                    momentum_rate /= 2.0
            else:
                if is_init: init_decreases += 1
                self.model.write_weights(''.join([self.output_name, '_best_weights']))
                if init_decreases == 15:
                    print "Tried to find initial learning rate, but failed, quitting"
                    break
                if not is_init: num_decreases += 1 #don't count num_decreases when trying to find initial learning rate
                print "cross-entropy did not decrease, so using previous best weights"
                self.model.open_weights(''.join([self.output_name, '_best_weights']))
                if num_decreases > 5: break
                learning_rate /= 2.0
                momentum_rate /= 2.0
#            sys.stdout.write("\r100.0% done \r")
#            sys.stdout.write("\r                                                                \r") #clear line           
            print "Epoch finished at", datetime.datetime.now()
        self.model.write_weights(self.output_name)
        end_time = datetime.datetime.now()
        print "Training finished at", end_time, "and ran for", end_time - start_time


def init_arg_parser():
    required_variables = dict()
    all_variables = dict()
    required_variables['train'] = ['feature_file_name', 'output_name', 'single_prediction']
    all_variables['train'] = required_variables['train'] + ['label_file_name', 'num_hiddens', 'weight_matrix_name', 
                               'save_each_epoch', 'backprop_batch_size',
                               'steepest_learning_rate', 'momentum_rate',
                               'validation_feature_file_name', 'validation_label_file_name',
                               'use_maxent', 'seed']
    required_variables['test'] =  ['feature_file_name', 'weight_matrix_name', 'output_name', 'single_prediction']
    all_variables['test'] = required_variables['test'] + ['label_file_name']
    required_variables['test'] =  ['feature_file_name', 'weight_matrix_name', 'output_name']
    all_variables['test'] =  required_variables['test'] + ['label_file_name']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='mode for DNN, either train or test', required=False)
    parser.add_argument('--config_file', help='configuration file to read in you do not want to input arguments via command line', required=False)
    for argument in all_variables['train']:
        parser.add_argument('--' + argument, required=False)
    for argument in all_variables['test']:
        if argument not in all_variables['train']:
            parser.add_argument('--' + argument, required=False)
    return parser

if __name__ == '__main__':
    #script_name, config_filename = sys.argv
    #print "Opening config file: %s" % config_filename
    script_name = sys.argv[0]
    parser = init_arg_parser()
    config_dictionary = vars(parser.parse_args())
    
    if config_dictionary['config_file'] != None :
        config_filename = config_dictionary['config_file']
        print "Since", config_filename, "is specified, ignoring other arguments"
        try:
            config_file=open(config_filename)
        except IOError:
            print "Could open file", config_filename, ". Usage is ", script_name, "<config file>... Exiting Now"
            sys.exit()
        
        del config_dictionary
        
        #read lines into a configuration dictionary, skipping lines that begin with #
        config_dictionary = dict([line.replace(" ", "").strip(' \n\t').split('=') for line in config_file 
                                  if not line.replace(" ", "").strip(' \n\t').startswith('#') and '=' in line])
        config_file.close()
    else:
        #remove empty keys
        config_dictionary = dict([(arg,value) for arg,value in config_dictionary.items() if value != None])

    try:
        mode=config_dictionary['mode']
    except KeyError:
        print 'No mode found, must be train or test... Exiting now'
        sys.exit()
    else:
        if (mode != 'train') and (mode != 'test'):
            print "Mode", mode, "not understood. Should be either train or test... Exiting now"
            sys.exit()
    
    if mode == 'test':
        test_object = RNNLM_Neural_Network_Addressee_Tester(config_dictionary)
    else: #mode ='train'
        train_object = RNNLM_Neural_Network_Addressee_Trainer(config_dictionary)
        train_object.backprop_steepest_descent_multi_batch_semi_newbob()
#        train_object.backprop_steepest_descent_single_batch_semi_newbob()
#        train_object.backprop_steepest_descent_single_batch()
        
    print "Finished without Runtime Error!" 
    