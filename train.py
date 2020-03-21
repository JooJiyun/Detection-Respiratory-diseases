import pandas as pd
import imageio
import os
import time
import tensorflow as tf
from tensorflow.contrib import rnn
import warnings
import random
import cv2
warnings.filterwarnings('ignore')

all_path = '../dataset/dataset_csv.csv'
diagnosis = ['URTI','Healthy','COPD','Asthma','LRTI','Bronchiectasis','Pneumonia','Bronchiolitis']
csv_path = ['../dataset/'+diagnosis[0]+'.csv',
            '../dataset/'+diagnosis[1]+'.csv',
            '../dataset/'+diagnosis[2]+'.csv',
            '../dataset/'+diagnosis[3]+'.csv',
            '../dataset/'+diagnosis[4]+'.csv',
            '../dataset/'+diagnosis[5]+'.csv',
            '../dataset/'+diagnosis[6]+'.csv',
            '../dataset/'+diagnosis[7]+'.csv']
IMG_WIDTH = 240
IMG_HIGHT = 177
NUM_CLASSES = 8 # diagnosis class
DATA_len = 5956
lr = 0.1

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

class DataManager():
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        self.current_data_offset = [0,0,0,0,0,0,0,0,0]

    def load_data(self):
        data = []
        target_data = []
        
        for i in range(self.batch_size):
            diag_idx = random.randrange(0,NUM_CLASSES)
            csv_file = pd.read_csv(csv_path[diag_idx])
            index = self.current_data_offset[diag_idx]
            if index == len(csv_file):
                index = 0
            
            data_img = cv2.imread(csv_file.loc[index, 'crop_path'])
            data.append(data_img)
            
            target_data.append([])
            for diag in diagnosis:
                target_data[i].append(int(csv_file.loc[index,diag]))
            index+=1
        self.current_data_offset[diag_idx] = index

        return data, target_data

    
class CRNN():
    def __init__(self, batch_size, model_path, restore):
        self.step = 0
        self.model_path = model_path
        self.save_path = os.path.join(model_path, 'ckp')

        self.restore = restore

        self.training_name = str(int(time.time()))
        self.session = tf.Session()

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
            ) = self.crnn(batch_size)
            self.init.run()

        with self.session.as_default():
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            if self.restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.saver.restore(self.session, ckpt)

        self.data_manager = DataManager(batch_size)

    def crnn(self, batch_size):
        def BidirectionnalRNN(inputs):
            
            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)# Forward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)# Backward

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, dtype=tf.float32)
                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)# Forward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)# Backward

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, dtype=tf.float32)
                outputs = tf.concat(outputs, 2)

            return outputs

        def CNN(inputs):
            
            with tf.variable_scope(None, default_name = 'CNN'):
                inputs = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "valid", activation=tf.nn.relu)
                print('*** conv1 ',inputs.shape,' ***')
                inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
                print('*** pool1 ',inputs.shape,' ***')
                inputs = tf.layers.conv2d(inputs=inputs, filters = 128, kernel_size = (3, 3), padding = "valid", activation=tf.nn.relu)
                print('*** conv2 ',inputs.shape,' ***')
                inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
                print('*** pool2 ',inputs.shape,' ***')
                inputs = tf.layers.conv2d(inputs=inputs, filters = 256, kernel_size = (3, 3), padding = "valid", activation=tf.nn.relu)
                print('*** conv3 ',inputs.shape,' ***')
                inputs = tf.layers.batch_normalization(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters = 256, kernel_size = (3, 3), padding = "valid", activation=tf.nn.relu)
                print('*** conv4 ',inputs.shape,' ***')
                inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=[1, 2])
                print('*** pool3 ',inputs.shape,' ***')
                inputs = tf.layers.conv2d(inputs=inputs, filters = 512, kernel_size = (3, 3), padding = "valid", activation=tf.nn.relu)
                print('*** conv5 ',inputs.shape,' ***')
                inputs = tf.layers.batch_normalization(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
                print('*** conv6 ',inputs.shape,' ***')
                inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=[1, 2])
                print('*** pool4 ',inputs.shape,' ***')
                inputs = tf.layers.conv2d(inputs=inputs, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)
                print('*** conv7 ',inputs.shape,' ***')
            
            return inputs

        inputs = tf.placeholder(tf.float32, [batch_size, IMG_HIGHT, IMG_WIDTH, 3])
        targets = tf.placeholder(tf.int32, [batch_size, NUM_CLASSES], name='targets')

        CNN_outputs = CNN(inputs)
        CNN_outputs = tf.reshape(CNN_outputs, [batch_size, -1, 512])
        print('*** cnn reshape ',CNN_outputs.shape,' ***')

        LSTM_outputs = BidirectionnalRNN(CNN_outputs)
        print('*** bilstm ',LSTM_outputs,' ***')
        logits = tf.reshape(LSTM_outputs, [batch_size, 363*512])
        print('*** logits ',logits.shape, ' ***')

        W1 = tf.Variable(tf.truncated_normal([363*512, 128], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0., shape=[128]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([128, NUM_CLASSES], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name="b2")

        logits = tf.matmul(logits, W1) + b1
        print('*** fc1 ',logits.shape,' ***')
        logits = tf.matmul(logits, W2) + b2

        #logits = tf.nn.sigmoid(logits)
        print('*** fc2 ',logits.shape,' ***')
        logits = tf.reshape(logits, [batch_size, NUM_CLASSES])
        print('*** fc reshape ',logits.shape,' ***')

        pred = tf.nn.softmax(logits)
        #cost = tf.losses.mean_squared_error(targets, logits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)


        correct_pred = tf.equal(tf.cast(tf.round(logits), tf.int32), targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        cost_hist = tf.summary.scalar('cost',cost)
        acc_hist = tf.summary.scalar('accuracy',accuracy)
        merged = tf.summary.merge_all()

        return inputs, targets, logits, optimizer, accuracy, cost, init, pred, merged

    def train(self, iteration_count):
        with self.session.as_default():
            print('Training')
            writer = tf.summary.FileWriter("./log/save_crnn_tb/", self.session.graph)
            for i in range(self.step, iteration_count + self.step):
                if i>50: lr = 0.05
                elif i>800: lr = 0.01
                elif i>1800: lr = 0.001
                batch_x, batch_y = self.data_manager.load_data()
                __, loss_value, acc_value, summ = self.session.run([self.optimizer, self.cost, self.acc, self.merged],
                    feed_dict={self.inputs: batch_x, self.targets: batch_y})
                
                self.saver.save( self.session, self.save_path, global_step=self.step )
                writer.add_summary(summ, i)
                print('[{}] Iteration loss: {}, accuracy : {}'.format(self.step, loss_value, acc_value))

                self.step += 1

                if i%10==0:
                    result, prediction = self.session.run([self.logits,self.pred], feed_dict={self.inputs: batch_x})
                    for k in range(10):
                        print('target : ',end = "")
                        for j, diag in enumerate(diagnosis):
                            print(batch_y[k][j],', ', end="")
                        print('')
                    for k in range(10):
                        print(' softmax : ', end="")
                        for j, diag in enumerate(diagnosis):
                            print('%0.1f'%prediction[k][j],', ', end="")
                        print('')
                    for k in range(10):
                        print('  prediction : ', end="")
                        for j, diag in enumerate(diagnosis):
                            print('%0.3f'%result[k][j],', ', end="")
                        print('')

        return None

    def test(self):
        prediction_csv = pd.DataFrame()
        with self.session.as_default():
            print('Testing')
            correct_n = 0
            for i in range(DATA_len):
                batch_x, batch_y = self.data_manager.load_data()
                result = self.session.run(self.pred,
                    feed_dict={self.inputs: batch_x})

                here_pred = 0
                for j, diag in enumerate(diagnosis):
                    prediction_csv.loc[i,'target_'+diag] = batch_y[0][j]
                    if batch_y[0][j]==1: here_target = diag
                for j, diag in enumerate(diagnosis):
                    prediction_csv.loc[i,diag] = result[0][j]
                    if here_pred<result[0][j]:
                        here_pred = result[0][j]
                        here_diag = diag
                for j, diag in enumerate(diagnosis):
                    if here_diag==diag:
                        prediction_csv.loc[i,'result_'+diag] = 1
                    else:
                        prediction_csv.loc[i,'result_'+diag] = 0
                if here_diag==here_target:
                    prediction_csv.loc[i,'correct'] = 1
                    correct_n+=1
                else:
                    prediction_csv.loc[i,'correct'] = 0

                if i%10==0:print('here : ',i)
        print('{}/{}'.format(correct_n, DATA_len))
        prediction_csv.to_csv('prediction_csv.csv',index = False)
        return None


crnn = CRNN(32,'./log/save_crnn/',False)
crnn.train(3000)