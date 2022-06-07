import pandas as pd
import os
import time
import tensorflow as tf
from tensorflow.contrib import rnn
import warnings
import random
import numpy as np
warnings.filterwarnings('ignore')

class DataManager():
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        self.current_train_offset = [0,0,0,0,0,0,0,0,0,0]
        self.current_test_offset = 0

    def train_data(self):
        data = []
        target_data = []
        
        for i in range(self.batch_size):
            diag_idx = random.randrange(0,NUM_CLASSES)
            index = self.current_train_offset[diag_idx]
            csv_file = pd.read_csv(csv_path[diag_idx])
            if index == len(csv_file):
                index = 0
            
            data_arr = csv_file.loc[index, 'mel']
            data_arr = np.load(data_arr)
            data.append(data_arr)
            
            
            target_data.append([])
            for diag in diagnosis:
                target_data[i].append(int(csv_file.loc[index,diag]))
            index+=1
            self.current_train_offset[diag_idx] = index

        return data, target_data

    def test_data(self):
        data = []
        target_data = []

        csv_file = pd.read_csv(all_path)
        for diag in out_diag:
            csv_file = csv_file[csv_file[diag]==0]
        csv_file = csv_file.reset_index(drop=True)

        index = self.current_test_offset
        for i in range(self.batch_size):
            if index == len(csv_file):
                index = 0
            
            data_arr = csv_file.loc[index, 'mel']
            data_arr = np.load(data_arr)
            data.append(data_arr)
            
            target_data.append([])
            for diag in diagnosis:
                target_data[i].append(int(csv_file.loc[index,diag]))
            index+=1
        self.current_test_offset = index

        return data, target_data

    
class Model():
    def __init__(self, batch_size, model_path, restore, test_saver, base_lr):
        self.step = 0
        self.model_path = model_path
        self.save_path = os.path.join(model_path, 'ckp')
        self.test_saver = test_saver
        self.restore = restore
        self.session = tf.Session()
        self.base_lr = base_lr
        self.batch_size = batch_size

        # Building graph
        with self.session.as_default():
            (
                self.inputs,
                self.targets,
                self.logits,
                self.optimizer,
                self.acc,
                self.cost,
                self.init,
                self.pred,
                self.merged
            ) = self.model(batch_size)
            self.init.run()

        with self.session.as_default():
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            self.saver_best = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            if self.restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.saver.restore(self.session, ckpt)


        self.data_manager = DataManager(batch_size)

    def restore_best(self):
        with self.session.as_default():
            ckpt = os.path.join(self.model_path, 'best_score')
            if ckpt:
                print('best score checkpoint is valid')
                self.saver.restore(self.session, ckpt)

    def model(self, batch_size):

        def RNN(inputs):
            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                gru_fw_cell_1 = tf.nn.rnn_cell.GRUCell(256)# Forward
                gru_bw_cell_1 = tf.nn.rnn_cell.GRUCell(256)# Backward

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell_1, gru_bw_cell_1, inputs, dtype=tf.float32)
                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                gru_fw_cell_2 = tf.nn.rnn_cell.GRUCell(256)# Forward
                gru_bw_cell_2 = tf.nn.rnn_cell.GRUCell(256)# Backward

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell_2, gru_bw_cell_2, inter_output, dtype=tf.float32)
                outputs = tf.concat(outputs, 2)

            return outputs

        def general_conv2d(inputs, o_d=64, f_w=3, s_w=1, stddev=0.02,
                   padding="valid", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
            with tf.variable_scope(name):

                conv = tf.contrib.layers.conv2d(
                    inputs, o_d, f_w, s_w, padding='same',
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=stddev),
                    biases_initializer=tf.constant_initializer(0.0)
                )
                if do_relu:
                    conv = tf.nn.relu(conv, "relu")
            return conv
        def build_resnet_block(inputs, idim, odim, projection, name, k_size, reduce = False):
            with tf.variable_scope(name):
                shorcut = inputs
                if projection:
                    shorcut = general_conv2d(shorcut, odim, k_size, 1, 0.02, name = "shortcut")
                inputs = general_conv2d(inputs, idim, k_size, 1, 0.02, name = "conv1")
                inputs = tf.contrib.layers.batch_norm(inputs)
                inputs = general_conv2d(inputs, idim, k_size, 1, 0.02, name = "conv2", do_relu=False)
                inputs = tf.contrib.layers.batch_norm(inputs)
                inputs = general_conv2d(inputs, odim, k_size, 1, 0.02, name = "conv3", do_relu=False)
                inputs = tf.contrib.layers.batch_norm(inputs)
                
            if reduce: 
            	return tf.layers.max_pooling2d(inputs=(tf.nn.relu(inputs+shorcut)), pool_size=[2, 2], strides=2)
            return tf.nn.relu(inputs + shorcut)
        def CNN(inputs):
            
            with tf.variable_scope(None, default_name = 'CNN'):
                inputs = general_conv2d(inputs, 1, 16, 1, 0.02, name="first_conv1")
                inputs = tf.contrib.layers.batch_norm(inputs)
                inputs = tf.nn.relu(inputs)
                inputs = tf.contrib.layers.max_pool2d(inputs, [2, 2], stride = 1) #112x112x3
                print('*** first_conv1 ',inputs.shape,' ***')

                inputs = build_resnet_block(inputs, 16, 32, True, "conv_block1", k_size = 2, reduce = True)
                print('*** conv_block1 ',inputs.shape,' ***')
                inputs = build_resnet_block(inputs, 16, 32, False, "identity1", k_size = 2, reduce = True)
                print('*** identity1 ',inputs.shape,' ***')
                inputs = build_resnet_block(inputs, 32, 64, True, "conv_block2", k_size = 3, reduce = True)
                print('*** conv_block2 ',inputs.shape,' ***')
                inputs = build_resnet_block(inputs, 32, 64, False, "identity2", k_size = 3, reduce = True)
                print('*** identity2 ',inputs.shape,' ***')
                inputs = build_resnet_block(inputs, 128, 512, True, "conv_block3", k_size = 5, reduce = True)
                print('*** conv_block3 ',inputs.shape,' ***')
                inputs = build_resnet_block(inputs, 128, 512, False, "identity3", k_size = 5)
                print('*** identity3 ',inputs.shape,' ***')
            
            return inputs

        inputs = tf.placeholder(tf.float32, [batch_size, IMG_HIGHT, IMG_WIDTH], name='inputs')
        targets = tf.placeholder(tf.int32, [batch_size, NUM_CLASSES], name='targets')
        inputs_re = tf.reshape(inputs, [batch_size, IMG_HIGHT, IMG_WIDTH, 1], name = 'inputs_re')

        CNN_outputs = CNN(inputs_re)
        print('*** cnn output ',CNN_outputs.shape,' ***')
        CNN_outputs = tf.reshape(CNN_outputs, [batch_size, -1, 512])
        print('*** cnn reshape ',CNN_outputs.shape,' ***')

        RNN_outputs = RNN(CNN_outputs)
        print('*** rnn output ',RNN_outputs.shape,' ***')
        logits = tf.reshape(RNN_outputs, [batch_size, 21*512])
        print('*** rnn reshape ',RNN_outputs.shape,' ***')

        W1 = tf.Variable(tf.truncated_normal([21*512, 512], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0., shape=[512]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([512, 128], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0., shape=[128]), name="b2")
        W3 = tf.Variable(tf.truncated_normal([128, NUM_CLASSES], stddev=0.1), name="W3")
        b3 = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name="b3")

        logits = tf.matmul(logits, W1) + b1
        print('*** fc1 ',logits.shape,' ***')
        logits = tf.matmul(logits, W2) + b2
        print('*** fc1 ',logits.shape,' ***')
        logits = tf.matmul(logits, W3) + b3

        #logits = tf.nn.sigmoid(logits)
        print('*** fc2 ',logits.shape,' ***')
        logits = tf.reshape(logits, [batch_size, NUM_CLASSES])
        print('*** fc reshape ',logits.shape,' ***')

        pred = tf.nn.softmax(logits)
        #cost = tf.losses.mean_squared_error(targets, logits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
        


        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(base_lr, global_step,100000, 0.96, staircase=True)

        decay_steps = 500
        lr_decayed = tf.train.cosine_decay(base_lr, global_step, decay_steps)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr_decayed).minimize(cost, global_step = global_step)

        correct_pred = tf.equal(tf.cast(tf.round(logits), tf.int32), targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        cost_hist = tf.summary.scalar('cost',cost)
        acc_hist = tf.summary.scalar('accuracy',accuracy)
        
        merged = tf.summary.merge_all()

        

        return inputs, targets, logits, optimizer, accuracy, cost, init, pred, merged

    def train(self, iteration_count):
        with self.session.as_default():
            compare = 0
            print('Training')
            writer = tf.summary.FileWriter(self.model_path+"_tb/", self.session.graph)
            for i in range(self.step, iteration_count+self.step):
                print('[{}] Iteration '.format(self.step))
                batch_x, batch_y = self.data_manager.train_data()
                __, loss_value, acc_value, summ = self.session.run([self.optimizer, self.cost, self.acc, self.merged],
                    feed_dict={self.inputs: batch_x, self.targets: batch_y})
                print('loss: {}, accuracy : {}'.format(loss_value, acc_value))
                
                self.saver.save( self.session, self.save_path, global_step=self.step )

                if compare<self.test(False):
                    self.saver_best.save( self.session, os.path.join(self.model_path, 'best_score'))
                    compare = self.test(True)


                writer.add_summary(summ, i)

                self.step += 1

        return None

    def test(self, csv_store = False):
        prediction_csv = pd.read_csv(all_path)
        for diag in out_diag:
            prediction_csv = prediction_csv[prediction_csv[diag]==0]
        prediction_csv = prediction_csv.reset_index(drop=True)
        prediction_csv = prediction_csv[diagnosis]

        with self.session.as_default():
            print('Testing')
            correct_n = 0
            compare_index = 0
            self.data_manager.current_test_offset = 0

            for i in range(round(DATA_len/self.batch_size +0.5)):
                batch_x, batch_y = self.data_manager.test_data()
                result = self.session.run(self.pred, feed_dict={self.inputs: batch_x})
                here_pred = 0
                for k in range(self.batch_size):
                    if compare_index>=DATA_len: break

                    for j, diag in enumerate(diagnosis):
                        if here_pred<result[k][j]:
                            here_pred = result[k][j]
                            here_diag = diag
                    for j, diag in enumerate(diagnosis):
                        if here_diag==diag:
                            prediction_csv.loc[compare_index,'result_'+diag] = 1
                        else:
                            prediction_csv.loc[compare_index,'result_'+diag] = 0

                    if prediction_csv.loc[compare_index, here_diag]==1:
                        prediction_csv.loc[compare_index,'correct'] = 1
                        correct_n+=1
                    else:
                        prediction_csv.loc[compare_index,'correct'] = 0
                    compare_index+=1

        print('{}/{}'.format(correct_n, DATA_len))
        if not csv_store : 
            return correct_n
        prediction_csv.to_csv(self.test_saver,index = False)
        for diag in diagnosis:
            print('-- ', diag)
            print('target : ', len(prediction_csv[prediction_csv[diag]==1]))
            print('TP : ', len(prediction_csv[(prediction_csv[diag]==1)&(prediction_csv['result_'+diag]==1)]))
            print('FP : ', len(prediction_csv[(prediction_csv[diag]==0)&(prediction_csv['result_'+diag]==1)]))
            print('TN : ', len(prediction_csv[(prediction_csv[diag]==0)&(prediction_csv['result_'+diag]==0)]))
            print('FN : ', len(prediction_csv[(prediction_csv[diag]==1)&(prediction_csv['result_'+diag]==0)]))

        return correct_n

# mfcc size:  (20, 862)
# stft size:  (276, 702)
# mel size:  (40, 702)

IMG_HIGHT = 40
IMG_WIDTH = 702
all_path = '../data/dataset.csv'

diagnosis = ['COPD','URTI','Asthma','Healthy','LRTI','Bronchiectasis','Pneumonia','Bronchiolitis']
out_diag = []
NUM_CLASSES = len(diagnosis)
csv_path = []
for diag in diagnosis:
    csv_path.append('../data/'+diag+'.csv')
DATA_len = 3048 
#DATA_len = 2069

base_lr = 0.01
MODEL = Model(16,'./log/gru_stft_7s_0s/',False, 'ignore.csv', base_lr)
MODEL.train(3000)
#MODEL.restore_best()
#MODEL.test(True)