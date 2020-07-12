import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from tensorflow.keras.optimizers import Adam
from tqdm import tqdm_notebook
from colorama import Fore, Style
def lrn(x, radius=2, alpha=1e-04, beta=0.75, bias = 1.0):
    return tf.nn.layers.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                                    beta = beta, bias = bias)
class Alexnet(tf.keras.Model):
    def __init__(self,
                 input_dim = (32, 32, 3),
                 out_dim = 10,
                 learning_rate = 1e-3,
                 checkpoint_directory = "checkpoints/",
                 device_name = "cpu:0"):
        super(Alexnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.lr = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name
        
        #layers
        self.conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                                     padding="same", activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), stride=(2, 2))
        self.conv2 = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1),
                                     padding="same", activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3), stride=(2, 2), padding="same")
        self.conv3 = Conv2D(filters=72, kernel_size=(3, 3), strides=(1, 1),
                                     padding="same", activation='relu')
        self.conv4 = Conv2D(filters=72, kernel_size=(3, 3), strides=(1, 1),
                                     padding="same", activation='relu')
        self.conv5 = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1),
                                     padding="same", activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(3, 3), stride=(2, 2), padding="same")
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.5)
        
        self.dense2 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)
        
        self.out = Dense(self.out_dim)
        
        #Optimizer
        self.optimizer = Adam(learning_rate = self.lr)
        
        self.global_step = 0
    
    #predict
    def predict(self, X, training):
        x = self.conv1(X)
        x = lrn(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x= lrn(x)
        x= self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropuot1(x)
            
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
            
        x = self.out(x)
        
        return x
    
    def call(self, X, training):
        return self.predict(X, training)
    
    def loss(self, X, y, training):
        
        prediction = self.predict(X, training)
        loss_val = tf.losses.sparse_softmax_cross_entropy(labels = y, logits=prediction)
        
        return loss_val, prediction
    
    def grad(self, X, y, training):
        
        with tfe.GradientTape() as tape:
            loss_val, pred = self.loss(X, y, training)
        return tape.gradient(loss_val, self.variables), loss_val
    #fit-for training
    def fit(self, X_train, y_train, X_val, y_val, epochs=1,
            verbose=1, batch_size=32, saving=False, tqdm_opyion=None):
        def tqdm_wrapper(*args, **kwargs):
            if tqdm_option == "normal":
                return tqdm(*args, **kwargs)
            elif tqdm_option == "notebook":
                return tqdm_notebook(*args, **kwargs)
            else:
                return args[0]
            
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(999999999).batch(batch_size)
        batch_trainlen = (len(X_train) - 1) // batch_size + 1
        
        dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(999999999).batch(batch_size)
        batch_vallen = (len(X_val) - 1) // batch_size + 1
        
        with tf.device(self.device_name):
            for i in range(epochs):
                epoch_loss = 0.0
                self.global_step +=1
                for X, y in tqdm_wrapper(dataset_train, total = batch_trainlen, desc="GLOBAL %s" % self.global_step):
                    grads, batch_loss = self.grad(X, y, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))
                epoch_loss_val = 0.0
                val_accuacy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_val, total = batch_vallen, desc="GLOBAL %s" % self.global_step):
                    batch_loss, pred =self.loss(X, y, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)
                    val_accuracy(tf,argmax.argmax(pred, axis=1), y)
                
                if i==0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i+1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss / batch_trainlen))
                    print("VAL   loss   : %.4f" % (epoch_loss_val / batch_vallen))
                    print("VAL   acc    : %.4f%%" % (val_accuarcy.result().numpy() * 100))
                    
                    if saving:
                        self.save()
                    print("=" * 25 + Style.RESET_ALL)
                time.sleep(1)
            
    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step = self.global_step)
        print("saved step %d in %s" % (self.gloabal_step,self.checkpoint_directory))
    #load your latest saved model   
    def load(slef, global_sep="latest"):
        dummy_input = tf.constant(tf.zeros((1,) + self.input_dim))
        dummpy_pred = self.call(dummy_input, True)
        
        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(sself.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
