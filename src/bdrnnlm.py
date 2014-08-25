'''
Created on Aug 22, 2014

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
from Bidirectional_RNNLM_Weight import Bidirectional_RNNLM_Weight
from Bidirectional_Recurrent_Neural_Network_Language_Model import Bidirectional_Recurrent_Neural_Network_Language_Model


class BDRNNLM_Tester(Bidirectional_Recurrent_Neural_Network_Language_Model): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(BDRNNLM_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.model.open_weights(self.weight_matrix_name)
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string',error_string="No label_file_name defined, just running forward pass",exit_if_no_default=False)
        if self.label_file_name is not None:
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

class BDRNNLM_Trainer(Bidirectional_Recurrent_Neural_Network_Language_Model):
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
        super(BDRNNLM_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.batch_size(self.feature_sequence_lens)
        self.num_sequences = self.features.shape[1]
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name is not None:
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

        if self.weight_matrix_name is not None:
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
    
    def calculate_gradient_single_batch(self, batch_inputs, batch_labels, gradient_weights, hiddens_forward = None, hiddens_backward = None, 
                                        outputs = None, check_gradient=False, model=None, l2_regularization_const = 0.0, 
                                        return_cross_entropy = False): 
        #need to check regularization
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        batch_size = batch_labels.size
        if model is None:
            model = self.model
        if hiddens_forward is None or hiddens_backward is None or outputs is None:
            outputs, hiddens_forward, hiddens_backward = self.forward_pass_single_batch(batch_inputs, model, return_hiddens=True)
        #derivative of log(cross-entropy softmax)
        batch_indices = np.arange(batch_size)
        gradient_weights *= 0.0
        backward_inputs = outputs

        if return_cross_entropy:
            cross_entropy = -np.sum(np.log2(backward_inputs[batch_indices, batch_labels]))
        backward_inputs[batch_indices, batch_labels] -= 1.0

        np.sum(backward_inputs, axis=0, out = gradient_weights.bias['output'][0])
        
        np.dot(hiddens_forward.T, backward_inputs, out = gradient_weights.weights['hidden_output_forward'])
        pre_nonlinearity_hiddens_forward = np.dot(backward_inputs[batch_size-1,:], model.weights['hidden_output_forward'].T) 
        pre_nonlinearity_hiddens_forward *= hiddens_forward[batch_size-1,:] 
        pre_nonlinearity_hiddens_forward *= 1 - hiddens_forward[batch_size-1,:]
        
        np.dot(hiddens_backward.T, backward_inputs, out = gradient_weights.weights['hidden_output_backward'])
        pre_nonlinearity_hiddens_backward = np.dot(backward_inputs[0,:], model.weights['hidden_output_backward'].T) 
        pre_nonlinearity_hiddens_backward *= hiddens_backward[0,:] 
        pre_nonlinearity_hiddens_backward *= 1 - hiddens_backward[0,:]
        
        if batch_size > 1:
            gradient_weights.weights['visible_hidden'][batch_inputs[batch_size-1]] += pre_nonlinearity_hiddens_forward
            gradient_weights.weights['hidden_hidden_forward'] += np.outer(hiddens_forward[batch_size-2,:], pre_nonlinearity_hiddens_forward)
            gradient_weights.bias['hidden_forward'][0] += pre_nonlinearity_hiddens_forward
            
            gradient_weights.weights['visible_hidden'][batch_inputs[0]] += pre_nonlinearity_hiddens_backward
            gradient_weights.weights['hidden_hidden_backward'] += np.outer(hiddens_backward[1,:], pre_nonlinearity_hiddens_backward)
            gradient_weights.bias['hidden_backward'][0] += pre_nonlinearity_hiddens_backward
        
        for index in range(batch_size-2):
            backward_index = batch_size - 2 - index
            forward_index = index + 1
            pre_nonlinearity_hiddens_forward = ((np.dot(backward_inputs[backward_index,:], model.weights['hidden_output_forward'].T) + 
                                                 np.dot(pre_nonlinearity_hiddens_forward, model.weights['hidden_hidden_forward'].T))
                                                * hiddens_forward[backward_index,:] * (1 - hiddens_forward[backward_index,:]))
            
            pre_nonlinearity_hiddens_backward = ((np.dot(backward_inputs[forward_index,:], model.weights['hidden_output_backward'].T) + 
                                                 np.dot(pre_nonlinearity_hiddens_backward, model.weights['hidden_hidden_backward'].T))
                                                * hiddens_backward[forward_index,:] * (1 - hiddens_backward[forward_index,:]))

            gradient_weights.weights['visible_hidden'][batch_inputs[backward_index]] += pre_nonlinearity_hiddens_forward #+= np.dot(visibles[observation_index,:,:].T, pre_nonlinearity_hiddens)
            gradient_weights.weights['hidden_hidden_forward'] += np.outer(hiddens_forward[backward_index-1,:], pre_nonlinearity_hiddens_forward)
            gradient_weights.bias['hidden_forward'][0] += pre_nonlinearity_hiddens_forward
            
            gradient_weights.weights['visible_hidden'][batch_inputs[forward_index]] += pre_nonlinearity_hiddens_backward #+= np.dot(visibles[observation_index,:,:].T, pre_nonlinearity_hiddens)
            gradient_weights.weights['hidden_hidden_backward'] += np.outer(hiddens_backward[forward_index+1,:], pre_nonlinearity_hiddens_backward)
            gradient_weights.bias['hidden_backward'][0] += pre_nonlinearity_hiddens_backward
        
        if batch_size > 1:
            pre_nonlinearity_hiddens_forward = ((np.dot(backward_inputs[0,:], model.weights['hidden_output_forward'].T) 
                                                 + np.dot(pre_nonlinearity_hiddens_forward, model.weights['hidden_hidden_forward'].T))
                                                * hiddens_forward[0,:] * (1 - hiddens_forward[0,:]))
            pre_nonlinearity_hiddens_backward = ((np.dot(backward_inputs[-1,:], model.weights['hidden_output_backward'].T) 
                                                  + np.dot(pre_nonlinearity_hiddens_backward, model.weights['hidden_hidden_backward'].T))
                                                * hiddens_backward[-1,:] * (1 - hiddens_backward[-1,:]))
            
        gradient_weights.weights['visible_hidden'][batch_inputs[0]] += pre_nonlinearity_hiddens_forward# += np.dot(visibles[0,:,:].T, pre_nonlinearity_hiddens)
        gradient_weights.weights['hidden_hidden_forward'] += np.outer(model.init_hiddens['forward'], pre_nonlinearity_hiddens_forward) #np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        gradient_weights.bias['hidden_forward'][0] += pre_nonlinearity_hiddens_forward
        gradient_weights.init_hiddens['forward'][0] = np.dot(pre_nonlinearity_hiddens_forward, model.weights['hidden_hidden_forward'].T)
        
        gradient_weights.weights['visible_hidden'][batch_inputs[-1]] += pre_nonlinearity_hiddens_backward# += np.dot(visibles[0,:,:].T, pre_nonlinearity_hiddens)
        gradient_weights.weights['hidden_hidden_backward'] += np.outer(model.init_hiddens['backward'], pre_nonlinearity_hiddens_backward) #np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        gradient_weights.bias['hidden_backward'][0] += pre_nonlinearity_hiddens_backward
        gradient_weights.init_hiddens['backward'][0] = np.dot(pre_nonlinearity_hiddens_backward, model.weights['hidden_hidden_backward'].T)
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
            gradient_weights *= batch_size
            if l2_regularization_const > 0.0:
                gradient_weights += model * (l2_regularization_const * batch_size)
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = Bidirectional_RNNLM_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = Bidirectional_RNNLM_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
            print "at initial hiddens"
            for key in direction.init_hiddens.keys():
                for index in range(direction.init_hiddens[key].size):
                    direction.init_hiddens[key][0][index] = epsilon
                    forward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model + direction)[batch_indices, batch_labels]))
                    backward_loss = -np.sum(np.log(self.forward_pass_single_batch(batch_inputs, model = model - direction)[batch_indices, batch_labels]))
                    finite_difference_model.init_hiddens[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.init_hiddens[key][0][index] = 0.0
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
            
            print "calculated gradient for forward initial hiddens"
            print gradient_weights.init_hiddens['forward']
            print "finite difference approximation for forward initial hiddens"
            print finite_difference_model.init_hiddens['forward']
            
            print "calculated gradient for backward initial hiddens"
            print gradient_weights.init_hiddens['backward']
            print "finite difference approximation for backward initial hiddens"
            print finite_difference_model.init_hiddens['backward']
            
            print "calculated gradient for forward hidden bias"
            print gradient_weights.bias['hidden_forward']
            print "finite difference approximation for forward hidden bias"
            print finite_difference_model.bias['hidden_forward']
            
            print "calculated gradient for backward hidden bias"
            print gradient_weights.bias['hidden_backward']
            print "finite difference approximation for backward hidden bias"
            print finite_difference_model.bias['hidden_backward']
            
            print "calculated gradient for output bias"
            print gradient_weights.bias['output']
            print "finite difference approximation for output bias"
            print finite_difference_model.bias['output']
            
            print "calculated gradient for visible_hidden layer"
            print gradient_weights.weights['visible_hidden']
            print "finite difference approximation for visible_hidden layer"
            print finite_difference_model.weights['visible_hidden']
            print np.sum((finite_difference_model.weights['visible_hidden'] - gradient_weights.weights['visible_hidden']) ** 2)
            
            print "calculated gradient for hidden_hidden_forward layer"
            print gradient_weights.weights['hidden_hidden_forward']
            print "finite difference approximation for hidden_hidden_forward layer"
            print finite_difference_model.weights['hidden_hidden_forward']
            
            print "calculated gradient for hidden_hidden_backward layer"
            print gradient_weights.weights['hidden_hidden_backward']
            print "finite difference approximation for hidden_hidden_backward layer"
            print finite_difference_model.weights['hidden_hidden_backward']
            
            print "calculated gradient for hidden_output_forward layer"
            print gradient_weights.weights['hidden_output_forward']
            print "finite difference approximation for hidden_output_forward layer"
            print finite_difference_model.weights['hidden_output_forward']
            
            print "calculated gradient for hidden_output_backward layer"
            print gradient_weights.weights['hidden_output_backward']
            print "finite difference approximation for hidden_output_backward layer"
            print finite_difference_model.weights['hidden_output_backward']
            
            sys.exit()
        ##########################################################
        
    def calculate_classification_statistics(self, features, flat_labels, feature_sequence_lens, model=None):
        if model is None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': [], 'init_hiddens' : []}
        
        batch_index = 0
        cross_entropy = 0.0
        log_perplexity = 0.0
        num_correct = 0
        start_frame = 0
        num_sequences = features.shape[1]
        num_outs = model.weights['hidden_output_forward'].shape[1]
        num_examples = self.batch_size(feature_sequence_lens)
        
        output = np.zeros((num_examples, num_outs))
#        print features.shape

        for batch_index, feature_sequence_len in enumerate(feature_sequence_lens):
            per_done = float(batch_index)/num_sequences*100
            sys.stdout.write("\r                                                                \r") #clear line
            sys.stdout.write("\rCalculating Classification Statistics: %.1f%% done " % per_done), sys.stdout.flush()
            end_frame = start_frame + feature_sequence_len
            batch_features = features[:feature_sequence_len, batch_index]
            batch_labels = flat_labels[start_frame:end_frame]
            output[start_frame:end_frame,:] = self.forward_pass_single_batch(batch_features, model, return_hiddens=False, linear_output=False)
            cross_entropy += self.calculate_cross_entropy(output[start_frame:end_frame,:], batch_labels)
            log_perplexity += self.calculate_log_perplexity(output[start_frame:end_frame,:], batch_labels)
            prediction = output[start_frame:end_frame,:].argmax(axis=1)
            num_correct += np.sum(prediction == batch_labels)
        
        sys.stdout.write("\r                                                                \r") #clear line
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        
#        cross_entropy /= np.log(2) * num_examples
        loss /= np.log(2) * num_examples
        log_perplexity /= num_examples
        perplexity = 2 ** log_perplexity
        return cross_entropy, perplexity, num_correct, num_examples, loss
    
    def backprop_steepest_descent_single_batch(self):
        print "Starting backprop using steepest descent"
        start_time = datetime.datetime.now()
        print "Training started at", start_time
        prev_step = Bidirectional_RNNLM_Weight()
        prev_step.init_zero_weights(self.model.get_architecture())
        gradient = Bidirectional_RNNLM_Weight()
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

    def backprop_adagrad_single_batch(self):
        print "Starting backprop using adagrad"
        adagrad_weight = Bidirectional_RNNLM_Weight()
        adagrad_weight.init_zero_weights(self.model.get_architecture())
        
        buffer_weight = Bidirectional_RNNLM_Weight()
        buffer_weight.init_zero_weights(self.model.get_architecture())
        
        fudge_factor = 1.0
        adagrad_weight = adagrad_weight + fudge_factor
        gradient = Bidirectional_RNNLM_Weight()
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
        