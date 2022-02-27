from sklearn.metrics import roc_auc_score,classification_report, roc_curve, confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
import numpy as np
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
import copy
from keras import backend as K

import pandas as pd
from time import time
from keras import applications, optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,ReLU, PReLU, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import tensorflow as tf

from numpy.random import seed
from tensorflow import set_random_seed
from functools import partial
from keras.layers import Concatenate, Lambda, Input

#specificity metric
def metric_specificity(y_true,y_pred):    
	TP=tf.reduce_sum(y_true*tf.round(y_pred))
	TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
	FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
	FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
	specificity=TN/(TN+FP+ 1e-8)
	return specificity

#sensitivity(recall) metric
def metric_recall(y_true,y_pred):  
	TP=tf.reduce_sum(y_true*tf.round(y_pred))
	TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
	FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
	FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
	recall=TP/(TP+FN+ 1e-8)
	return recall

#precision metric
def metric_precision(y_true,y_pred):    
	TP=tf.reduce_sum(y_true*tf.round(y_pred))
	TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
	FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
	FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
	precision=TP/(TP+FP+ 1e-8)
	return precision

#F1-score metric
def metric_F1score(y_true,y_pred):    
	TP=tf.reduce_sum(y_true*tf.round(y_pred))
	TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
	FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
	FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
	precision=TP/(TP+FP + 1e-8)
	recall=TP/(TP+FN + 1e-8)
	F1score=2*precision*recall/(precision+recall + 1e-8)
	return F1score

def thorough_numerical_evaluation(model,validation_generator,training_generator,threshold):
	probabilities = model.predict_generator(validation_generator, 1,verbose=1)
	score = model.evaluate_generator(validation_generator)
	
	true_classes = training_generator.classes
	predictions = model.predict_generator(generator = training_generator, steps = 2,workers=1)
	roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))
	print('Training AUC')
	print(roc_auc)

	true_classes = validation_generator.classes
	predictions = model.predict_generator(generator = validation_generator, steps = 2,workers=1)
	roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))
	print('Testing AUC')
	print(roc_auc)



	predictions = (predictions-min(predictions))/(max(predictions)-min(predictions)) #Normalize between 0 and 1.
	print(predictions)
	print(validation_generator.filenames)

	fpr, tpr, thresholds = roc_curve(true_classes, np.ravel(predictions))
	print('Specificity (1-FPR)')
	specificity = 1- fpr
	print(specificity)
	print('TPR (sensitivity)')
	print(tpr)
	print('Thresholds')
	print(thresholds)
	plt.figure()
	plt.tight_layout()
	plt.plot(fpr,tpr,color = 'darkorange',label = 'ROC curve (area = %0.2f)'% roc_auc)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
	plt.legend(loc="lower right")

	ax2 = plt.gca().twinx()
	ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
	ax2.set_ylabel('Threshold',color='r')
	ax2.set_ylim([thresholds[-1],thresholds[0]])
	ax2.set_xlim([fpr[0],fpr[-1]])

	plt.savefig('ROCcurve.eps')

	dmindex = predictions <= threshold
	nodmindex = predictions > threshold
	predictions[dmindex] = 0
	predictions[nodmindex] = 1
	print(predictions)
	cnf = confusion_matrix(true_classes,predictions)
	plot_confusion_matrix(cnf,['dm','nodm'],normalize=False)

	print(validation_generator.class_indices)
	print(validation_generator.classes)
	print(model.metrics_names)
	print(score)

class Histories(Callback):
	
	def __init__(self, validation_generator = None, train_generator = None):
		super(Histories, self).__init__()
		self.validation_generator = validation_generator
		self.train_generator = train_generator

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.trainingaucs = []
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		valid_steps = np.ceil(self.validation_generator.samples/self.validation_generator.batch_size)
		true_classes = self.validation_generator.classes
		predictions = self.model.predict_generator(generator = self.validation_generator, steps = valid_steps,workers=1);
		print(predictions[-30:])

		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.aucs.append(round(roc_auc,3))
		print('Validation AUCS')
		print(self.aucs)

		valid_steps = np.ceil(self.train_generator.samples/self.train_generator.batch_size)
		true_classes = self.train_generator.classes
		predictions = self.model.predict_generator(generator = self.train_generator, steps = valid_steps,workers=1)
		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.trainingaucs.append(round(roc_auc,3))
		print('Training AUCS')
		print(self.trainingaucs)

		return

class MultiGPUCheckpointCallback(Callback):

	def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super(MultiGPUCheckpointCallback, self).__init__()
		self.base_model = base_model
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.save_weights_only = save_weights_only
		self.period = period
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if self.verbose > 0:
							print('Epoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
						self.best = current
						if self.save_weights_only:
							self.base_model.save_weights(filepath, overwrite=True)
						else:
							self.base_model.save(filepath, overwrite=True)
					else:
						if self.verbose > 0:
							print('Epoch %05d: %s did not improve' %
								  (epoch + 1, self.monitor))
			else:
				if self.verbose > 0:
					print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
					self.base_model.save_weights(filepath, overwrite=True)
				else:
					self.base_model.save(filepath, overwrite=True)

def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x

def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='Block35'`
        - Inception-ResNet-B: `block_type='Block17'`
        - Inception-ResNet-C: `block_type='Block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals before adding
            them to the shortcut branch.
        block_type: `'Block35'`, `'Block17'` or `'Block8'`, determines
            the network structure in the residual branch.
        block_idx: used for generating layer names.
        activation: name of the activation function to use at the end
            of the block (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'Block35'`,
            `'Block17'` or `'Block8'`.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 48, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 64, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name_fmt('ScaleSum'))([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x

def build_a_new_model_3(inputShape = (512, 512, 1)):
	#model = Sequential() #Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power). 

	img_input = Input(shape=inputShape)
	x = conv2d_bn(img_input, 32, 5)
	# model.add(Conv2D(32, (5, 5), input_shape=inputShape)) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes. 
	# model.add(BatchNormalization())
	# model.add(ReLU())#PReLU())
	# x = _inception_resnet_block(x,
 #                                    scale=0.17,
 #                                    block_type='Block35',
 #                                    block_idx=1)
	x = MaxPooling2D(pool_size=(4, 4),strides=4)(x)

	# model.add(Conv2D(64, (3, 3))) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes. 
	# model.add(BatchNormalization())
	# model.add(ReLU())#PReLU())
	x = conv2d_bn(x, 32, 3)
	x = _inception_resnet_block(x,
                                    scale=0.17, #0.17,
                                    block_type='Block35',
                                    block_idx=2)
	x = MaxPooling2D(pool_size=(4, 4),strides=4)(x)

	# model.add(Conv2D(128, (3, 3)))
	# model.add(BatchNormalization())
	# model.add(ReLU())#PReLU())
	x = conv2d_bn(x, 64, 3)
	# x = _inception_resnet_block(x,
 #                                    scale=0.17,
 #                                    block_type='Block35',
 #                                    block_idx=3)
	x = MaxPooling2D(pool_size=(4, 4),strides=4)(x)

	x = Flatten()(x)#model.add(GlobalAveragePooling2D())
	x = Dense(128)(x)
	x = ReLU()(x)
	x = Dense(64)(x)
	x = ReLU()(x)#LeakyReLU())#PReLU())
	x = Dropout(0.25)(x)
	x = Dense(1)(x)
	x = Activation('sigmoid')(x)

	model = Model(img_input, x, name='inception_resnet_v2')

	return model

def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.binary_crossentropy(y_true, y_pred) #loss1 = K.categorical_crossentropy(y_true, y_pred)
    #loss2 = (-1) * K.mean(K.sum((K.ones_like(y_pred) -y_pred)*K.log( K.ones_like(y_pred)-y_pred)+y_pred*K.log(y_pred)))
    #loss2 = (-1) * K.sum((K.ones_like(y_pred) -y_pred)*K.log( K.ones_like(y_pred)-y_pred)+y_pred*K.log(y_pred), axis=1)
    loss2 = (-1) * ((K.ones_like(y_pred) -y_pred)*K.log( K.ones_like(y_pred)-y_pred+K.epsilon())+y_pred*K.log(y_pred+K.epsilon()))
    #print(loss1)
    #print(loss2)
    #loss1 = K.print_tensor(loss1, message='loss1')
    #loss2 = K.print_tensor(loss2, message='loss2')
    #loss_all = K.print_tensor(loss1+loss2, message='loss_all')
    #loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return loss1 + loss2 #(1-e)*loss1 + e*loss2

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

def preprocess_input(x):
    
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def main():


	######## Forces the GPUs to not pre-hog all the memory.
	config = tf.ConfigProto() 
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	K.set_session(sess)

	######## Initialize directories.
	rootdir = "/home/shuchao/codes/CTcnn/CT/"
	imagedir = rootdir + "IMAGES_LC_CENTER/"
	outcometype = 'LR/'
	#logtitle = "2ndRun_2020"
	logtitle = 'released_ours_4_newNet_2_newAUC_united_2_LR'
	size = 512
	print(logtitle)

	######## Build the model.

	model = build_a_new_model_3((size,size,1))


	######## If starting from previously computed weights, (very basic transfer learning or reinitializing model training).

	bestweights = "./models/" + outcometype + logtitle + "_weights.best.hdf5"

	model.load_weights(bestweights)

	# for layer in model.layers[:9]: #Example of how to freeze all layers up until layer 9, to only re-train the fully connected layers.
	# 	layer.trainable = False	
	# print("Layers frozen until " + model.layers[9].name)

	######################################
	#gpu_model = model #  print("Layers frozen until " + model.layers[9].name)
	gpu_model = multi_gpu_model(model, gpus = 2) #This is what allows the model to use both (or more GPUS). Note there is no sharing of memory, just linearly improves the computation time. 

	print(model.summary()) # Instantly generates full summary.

	#Initialize variables & print to log.
	lr = 0.001
	momentum = 0.5
	batch_size = 32
	print('lr = ' + str(lr))
	print('momentum = ' + str(momentum))
	print('batch size = ' + str(batch_size))

	#Compilation step. Needed to initialize the model, pick loss, optimizer, etc.. 
	gpu_model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], #mycrossentropy, #'binary_crossentropy',
				 #optimizer=optimizers.Adam(),
				  optimizer=optimizers.SGD(lr=lr,momentum=momentum),
				  metrics=['accuracy','mse','binary_accuracy', metric_specificity, metric_recall, metric_precision, metric_F1score])
	

	#Generation of input data. Two seperate generators for validation and train (as validation does not use data-augmentation)
	train_datagen = ImageDataGenerator(rescale=1. / 255,rotation_range=20, horizontal_flip= True, vertical_flip = True, width_shift_range = 0.4, height_shift_range = 0.4)
	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
			imagedir + 'TRAIN_imbalanced',  # this is the target directory
			target_size=(size, size),  
			batch_size=batch_size,
			class_mode='binary',
			color_mode = 'grayscale')  # since we use binary_crossentropy loss, we need binary labels

	train_generator_forauc = train_datagen.flow_from_directory(
			imagedir + 'TRAIN_imbalanced',  # this is the target directory
			target_size=(size, size),  
			batch_size=batch_size,
			class_mode='binary',
			color_mode = 'grayscale',
			shuffle = False)  # No need to shuffle as this is just to compute a metric after training. 

	validation_generator = validation_datagen.flow_from_directory(
			imagedir + 'TEST',
			target_size=(size, size),
			batch_size=106, 				#I've manually set this so the validation process uses two batches instead of one. (Irrelevant for numerical purposes, but can run into memory issues otherwise) 
			class_mode='binary',    
			color_mode = 'grayscale',
			shuffle = False)  # No need to shuffle as this is just to compute a metric after training. 

	classweights = {0 : 1, 1: 1} # If you wanted to perform any sort of imbalance adjustments, this is where it would be.


	valid_steps = np.ceil(validation_generator.samples/validation_generator.batch_size)
	probabilities = gpu_model.predict_generator(validation_generator, valid_steps,verbose=1,workers=1)
	print(probabilities)
	print(validation_generator.class_indices)
	print(validation_generator.classes)

	true_classes = validation_generator.classes
	roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(probabilities))

	print('AUC on testing set:', round(roc_auc,3))
	score = gpu_model.evaluate(validation_generator)
	print('Loss on testing set:', score[0])
	print('Accuracy on testing set:', score[1])
	print('MSE on testing set:', score[2])
	print('Binary_accuracy on testing set:', score[3])
	print('Specificity on testing set:', score[4])
	print('Sensitivity (recall) on testing set:', score[5])
	print('Precision on testing set:', score[6])
	print('F1_Score on testing set:', score[7])

	y_true = validation_generator.classes
	
	speci_max = 0
	speci_max_thres = 0.0
	recall_max = 0
	recall_max_thres = 0.0
	f1_max = 0
	f1_max_thres = 0.0
	preci_max = 0
	ba_max = 0

	for i in np.arange(0.0,1.0,0.0001):
		thres = i
		y_pred = copy.deepcopy(probabilities)
		dmindex = y_pred >= thres
		nodmindex = y_pred < thres
		y_pred[dmindex] = 1
		y_pred[nodmindex] = 0
		y_pred = y_pred.reshape(y_true.shape)
		TP=np.sum(y_true * (y_pred))
		TN=np.sum((1-y_true)*(1-(y_pred)))
		FP=np.sum((1-y_true)*(y_pred))
		FN=np.sum(y_true*(1-(y_pred)))
		specificity=TN/(TN+FP+ 1e-8)
		# if specificity > speci_max:
		# 	speci_max = specificity
		# 	speci_max_thres = thres

		recall=TP/(TP+FN+ 1e-8)
		# if recall > recall_max:
		# 	recall_max = recall
		# 	recall_max_thres = thres
		precision=TP/(TP+FP + 1e-8)
		#recall=TP/(TP+FN + 1e-8)
		F1score=2*precision*recall/(precision+recall + 1e-8)
		#if F1score > f1_max:
		ba = (specificity + recall) / 2.0
		if ba > ba_max:
			f1_max = F1score
			f1_max_thres = thres
			speci_max = specificity
			recall_max = recall
			preci_max = precision
			ba_max = ba

	print('based on {}, the Specificity is {}'.format(f1_max_thres,speci_max))
	print('based on {}, the Sensitivity (Recall) is {}'.format(f1_max_thres,recall_max))
	print('based on {}, the balanced accuracy is {}'.format(f1_max_thres, ba_max))
	print('based on {}, the Precision is {}'.format(f1_max_thres,preci_max))
	print('based on {}, the F1 Score is {}'.format(f1_max_thres,f1_max))

	

if __name__ == "__main__":
	x = main()
