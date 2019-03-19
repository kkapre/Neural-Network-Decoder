# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:00:30 2019

@author: kkapr
"""
from OFDM import OFDM_module
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
import numpy as np
import sk_dsp_comm.fec_conv as fec
import sk_dsp_comm.sigsys as ss
import scipy.signal as signal
import scipy.special as special
import sk_dsp_comm.digitalcom as dc
import sk_dsp_comm.fec_block as block
from keras import backend as K
from sklearn.model_selection import train_test_split
from numpy.random import seed
from scipy.interpolate import spline
#seed(1)
from tensorflow import set_random_seed
import pyldpc;


class NNdecoder():
    def __init__(self,k=4,encoder_type='hamming',channel_type='AWGN',mod_type='BPSK',
                 train_w_bitstream=True,Mblocks=50,p=1,
                 m_batch_size=10,m_epochs=100,m_loss='binary_crossentropy',
                 m_metric='bermetric',num_layers=np.array([128,64,32]),
                 EbN0low=-2,EbN0high=10,num_snr_tested=20,M=10000):
        self.k = 4;
        self.encoder_type = encoder_type;
        self.channel_type = channel_type;
        self.mod_type = mod_type;
        self.train_w_bitstream = train_w_bitstream;
        self.Mblocks = Mblocks;
        self.p = p;
        self.m_batch_size = m_batch_size;
        self.m_epochs = m_epochs;
        self.m_loss = m_loss;
        self.m_metric = m_metric;
        self.num_layers = num_layers;
        self.EbN0low = EbN0low;
        self.EbN0high = EbN0high;
        self.EbN0 = EbN0low;
        self.num_snr_tested = num_snr_tested;
        self.snr_range = np.linspace(self.EbN0low,self.EbN0high,self.num_snr_tested);
        self.snr_train_range = np.linspace(self.EbN0low,self.EbN0high,self.num_snr_tested);
        self.m_optimizer = 'adam';
        self.M = M;
        #self.createNNdecoder();
        self.x_test = np.empty(1);
        self.y_test = np.empty(1);
        self.myOFDM = OFDM_module(self.x_test);
        self.num_carriers = self.myOFDM.K;
        self.initEncoder();
        self.createData();
        if(channel_type=='OFDM'):
            #nn_input_size = 32;
            self.nn_input_size = 2*self.N
            self.nn_output_size = self.k;
            #zero_pad = True;
        else:
            self.nn_input_size = self.N;
            self.nn_output_size = self.k;        
# =============================================================================
# Neural Network
# =============================================================================
    def bermetric(self,x, y):
        return K.mean(K.not_equal(x, K.round(y)));
    def createNNdecoder(self):     
        self.model = Sequential()#create model
        self.model.add(Dense(self.num_layers[0],input_dim=self.nn_input_size, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(self.num_layers[1], kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(self.num_layers[2], kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(self.nn_output_size, kernel_initializer='uniform', activation='sigmoid'))
        if(self.m_metric == 'accuracy'):
            self.model.compile(optimizer=self.m_optimizer, loss=self.m_loss, metrics=['accuracy'])
        else:
            self.model.compile(optimizer=self.m_optimizer, loss=self.m_loss, metrics=[self.bermetric]);
        return self.model;      
    def train(self):
        history = self.model.fit(self.x_train,self.y_train,batch_size=self.m_batch_size,epochs=self.m_epochs,verbose=False)
        return history;
    def test(self):
        predictions = self.model.predict(self.x_test);# returns probabilities
        output = np.round(predictions);
        ber = self.BER(self.y_test,output);
        return output,ber;
# =============================================================================
# Encoder 
# =============================================================================
    def initEncoder(self):
        self.j=3;
        if(self.encoder_type=='hamming'):
            n_hamm = 2**self.j - 1;
            k_hamm = n_hamm-self.j
            r = k_hamm/n_hamm
            self.N = np.int(self.k/r)
        if(self.encoder_type=='ldpc'):
            self.k=6;
            self.N,d_v,d_c = 15,4,5;
            self.H = pyldpc.RegularH(self.N,d_v,d_c)
            self.Gt = pyldpc.CodingMatrix(self.H);            

    def hamming_encode(self,bits):
        bits = np.int32(bits);
        hh1 = block.fec_hamming(self.j);
        
        y = hh1.hamm_encoder(bits)
        return y
    def Encoder(self,X):
        X = np.int32(X);
        Xnumrows = np.shape(X)[0]
        if (self.encoder_type == 'hamming'):
            enc = self.hamming_encode(np.ravel(X))
            return np.reshape(enc,(Xnumrows,self.N));
    
        if(self.encoder_type=='ldpc'):
            Xt = np.transpose(X);
            Yt = np.matmul(self.Gt,Xt);
            Y = np.transpose(Yt);
            Y = np.remainder(Y,2);
            return Y;
# =============================================================================
# Modulation
# =============================================================================
    def Modulation(self,x):
        if(self.mod_type == 'BPSK'):
            symbols1 = 2*x-1;
            symbols1 = np.ravel(symbols1);
        return symbols1;
# =============================================================================
# System
# =============================================================================
    def channelOutput(self,x):
         #orig_shape = x.shape
        x_flat = np.ravel(x)
        if(self.channel_type=='OFDM'):
            symbols = self.Modulation(x_flat);
            orig_symb_size = np.size(symbols);
            symbols = self.fit_to_input(x,self.num_carriers) 
            symb_rows = symbols.shape[0];
            self.myOFDM = OFDM_module(symbols[0,:]);
            self.myOFDM.setSNR(self.EbN0)
            y = self.myOFDM.OFDM_run()
            for i in range(1,symb_rows):
                self.myOFDM = OFDM_module(symbols[i,:])
                self.myOFDM.setSNR(self.EbN0)
                y = np.append(y,self.myOFDM.OFDM_run())
                y = np.ravel(y);
                y = y[0:orig_symb_size];
        else:
             y = dc.cpx_AWGN(2*x_flat-1,self.EbN0+10*np.log10(self.k/self.N),1);
    
        if((self.channel_type=='OFDM')):
            return y;
        else:
            return y.real;
        

    def sendThroughCommSystem(self,msg):
        num_msg_bits = msg.size;
        encodable = self.fit_to_input(msg,self.k);
        enc_out = self.Encoder(encodable);
        #enc_out = np.int32(enc_out);
        enc_out = np.ravel(enc_out);
        chan_out = self.channelOutput(enc_out);
             
        if(self.channel_type=='OFDM'):
            nn_input = self.fit_to_input(chan_out,np.int(self.nn_input_size/2));
            nn_input_complex = np.concatenate((nn_input.real,nn_input.imag),axis=1);
            nn_input = nn_input_complex;
        else:
            nn_input = self.fit_to_input(chan_out,self.nn_input_size);
        nn_out = self.model.predict(nn_input);
        rec_msg = np.round(nn_out);
        rec_msg = np.ravel(rec_msg);
        rec_msg = rec_msg[0:num_msg_bits]
            
        return rec_msg;
# =============================================================================
# Run Training and Tests
# =============================================================================
    def trainOverSNRs(self,plot_Convergence=False):
        print("Training...",end="");
        self.nn_ber = np.zeros(self.num_snr_tested)
        for i in range(self.num_snr_tested):
            self.EbN0 = self.snr_range[i];#increase SNR each time
            nn_input_og = self.channelOutput(self.cwords);
            nn_input = self.fit_to_input(nn_input_og,self.d.shape[0],1)
            fin_nn_in = nn_input;
            if(self.channel_type=='OFDM'):
                nn_input_complex = np.concatenate((nn_input.real,nn_input.imag),axis=1);
                fin_nn_in = nn_input_complex;
    
    
            if(i==0):
                self.nn_input_size = fin_nn_in.shape[1]
                self.nn_output_size = self.d.shape[1];
                self.model = self.createNNdecoder()#initialize decoder
                self.init_weights = self.model.get_weights();#get initial weights
            if(self.p!=1):#if data is going to be split into train and test sets
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(fin_nn_in, self.d, test_size=1-self.p,random_state=42,shuffle=True)
            else:
    
                self.x_train, self.x_test, self.y_train, self.y_test = fin_nn_in,fin_nn_in,self.d,self.d;
               # x_train, x_test, y_train, y_test = train_test_split(nn_input, dfit, shuffle=True,train_size=0.9999,test_size=0.9999)
    
            if(i==0):
                self.y_training = self.y_train;
                self.y_training = self.y_training[:,:,np.newaxis]
                self.y_testing = self.y_test;
                self.y_testing = self.y_testing[:,:,np.newaxis]
                self.x_testing = self.x_test;
                self.x_testing = self.x_testing[:,:,np.newaxis]
            else:
                self.y_training = np.append(self.y_training,np.expand_dims(self.y_train,axis=2),axis=2)
                self.y_testing = np.append(self.y_testing,np.expand_dims(self.y_test,axis=2),axis=2)
                self.x_testing = np.append(self.x_testing,np.expand_dims(self.x_test,axis=2),axis=2)
            cwordhist = self.train();#train with new output data
            if(plot_Convergence):
                plt.figure(1);
                metric_key =list(cwordhist.history.keys())[1]#get string of metric used
                plt.plot(cwordhist.history[metric_key]);
                plt.xlabel('Epochs');
                plt.ylabel('BER')
                plt.title(f'BER vs Epochs for Trained Codewords')
        if(self.train_w_bitstream):
            #nn_msg_ber = np.zeros(num_snr_tested)
            #Mblocks = 100; #number of k bit blocks to train
            for i in range(self.num_snr_tested):
                self.EbN0 = self.snr_range[i];
                if(self.p!=1):
                    msg_stream = self.createPermutationsOf(self.y_training[:,:,i],self.Mblocks)
                    rand_msg = msg_stream;
                else:
                    rand_msg = np.random.randint(0,2,self.k*self.Mblocks);
                dstream = np.reshape(rand_msg,(self.Mblocks,self.k))
                encodable = self.fit_to_input(rand_msg,self.k);
                enc_out = self.Encoder(encodable);
                #enc_out = np.int32(enc_out);
                enc_out = np.ravel(enc_out);
                nn_input_og = self.channelOutput(enc_out);
                nn_input = self.fit_to_input(nn_input_og,dstream.shape[0],1)
                fin_nn_in = nn_input;
                if(self.channel_type=='OFDM'):
                    nn_input_complex = np.concatenate((nn_input.real,nn_input.imag),axis=1);
                    fin_nn_in = nn_input_complex;
                if(i==0):
                    self.nn_input_size = fin_nn_in.shape[1]
                    self.nn_output_size = dstream.shape[1];
                    #model1 = createNNdecoder()#initialize decoder
                    self. init_weights = self.model.get_weights();#get initial weights

                self.x_train, self.x_test, self.y_train, self.y_test = fin_nn_in,fin_nn_in,dstream,dstream;
                   # x_train, x_test, y_train, y_test = train_test_split(nn_input, dfit, shuffle=True,train_size=0.9999,test_size=0.9999)
    
                bitstreamhist =self.train();#train with new output data
                if(plot_Convergence):
                    plt.figure(2)
                    metric_key =list(bitstreamhist.history.keys())[1]#get string of metric used
                    plt.plot(bitstreamhist.history[metric_key]);
                    plt.xlabel('Epochs');
                    plt.ylabel('BER')
                    plt.title(f'BER vs Epochs for Random Bitstream of Length {self.Mblocks*self.k}')
                #nn_ber[i]=test()[1];
        
        
    def testOverSNRs(self):
        #test bitstream
        print("Testing...",end="");
        self.nn_msg_ber = np.zeros(self.num_snr_tested)
        for i in range(self.num_snr_tested):
            self.EbN0 = self.snr_range[i];
            rand_msg = np.random.randint(0,2,self.k*self.M);
            rec_msg = self.sendThroughCommSystem(rand_msg);
            self.nn_msg_ber[i] = self.BER(rand_msg,rec_msg);
    
    
        if(self.p!=1):
        #test bitstream made only of untested codewords
            self.nn_testmsg_ber = np.zeros(self.num_snr_tested)
            for i in range(self.num_snr_tested):
                self.EbN0 = self.snr_range[i];
                test_msg_stream = self.createPermutationsOf(self.y_testing[:,:,i],self.k*self.M)
    
                test_rec_msg = self.sendThroughCommSystem(test_msg_stream);
                self.nn_testmsg_ber[i] = self.BER(test_msg_stream,test_rec_msg);
    
    
    
            #test bitstream made only trained codewords
            self.nn_trainmsg_ber = np.zeros(self.num_snr_tested)
            for i in range(self.num_snr_tested):
                self.EbN0 = self.snr_range[i];
                train_msg_stream = self.createPermutationsOf(self.y_training[:,:,i],self.k*self.M)
                train_rec_msg = self.sendThroughCommSystem(train_msg_stream);
                self.nn_trainmsg_ber[i] = self.BER(train_msg_stream,train_rec_msg);  
        print("Complete");
        
    def plotBER(self):
        if(self.p!=1):
            #n1, = plt.plot(snr_range,nn_ber);
            
            n2, = plt.plot(self.snr_range,self.nn_msg_ber)
            m1, = plt.plot(self.snr_range,self.map_ber)
            n3, = plt.plot(self.snr_range,self.nn_testmsg_ber)
            n4, = plt.plot(self.snr_range,self.nn_trainmsg_ber)
             #plt.plot(snr_range,map_ber2)
            plt.yscale('log')
            plt.ylabel('BER')
            plt.xlabel('SNR(dB)')
            title = f"BER vs SNR for {self.channel_type} k={self.k} N={self.N} p={self.p} with {self.encoder_type} encoding, {self.num_snr_tested} SNRs trained, {self.m_epochs} Epochs"
            plt.title(title)
            plt.legend([n2,n3,n4,m1],['Neural Net Decoder','NN:Test Codewords','NN:Trained Codewords','MAP'])
    
        else:
            n2, = plt.plot(self.snr_range,self.nn_msg_ber)
            m1, = plt.plot(self.snr_range,self.map_ber)
            plt.yscale('log')
            plt.ylabel('BER')
            plt.xlabel('SNR(dB)')
            if(self.train_w_bitstream):
                title = f"BER vs SNR for {self.channel_type} k={self.k} N={self.N} w/{self.num_snr_tested} SNRs trained w/Bitstream of length {self.Mblocks*self.k}"
            else:
                 title = f"BER vs SNR for {self.channel_type} k={self.k} N={self.N} w/{self.num_snr_tested} SNRs trained w/o Bitstream"
            plt.title(title);
            plt.legend([n2,m1],['Neural Net Decoder','MAP'])
    def run(self,plot_Convergence=False):
        self.MAPdecodertest()
        self.trainOverSNRs(plot_Convergence);
        self.testOverSNRs();
    def MAPdecodertest(self,coded=True):    
        self.map_ber = block.block_single_error_Pb_bound(self.j,self.snr_range,coded);
# =============================================================================
# Helper Functions
# =============================================================================
    def BER(self,x,y):
        return np.mean(np.not_equal(x,y))
    def printParameters(self):
        if(self.train_w_bitstream):
            parameters = f'Channel is {self.channel_type}\n{self.encoder_type} encoder with k={self.k}, N=7\n{self.num_snr_tested} SNRs trained\nBitstream of length {self.Mblocks*self.k} is trained\n{self.num_layers} layer type,{self.m_epochs} epochs\np={self.p}'
        else:
            parameters = f'Channel is {self.channel_type}\n{self.encoder_type} encoder with k={self.k}, N=7\n{self.num_snr_tested} SNRs trained\nBitstream not used for training\n{self.num_layers} layer type,{self.m_epochs} epoch\np={self.p}'
        print(parameters);
    def fit_to_input(self,arr,in_len,axis=0):
        if(axis==0):
            if(arr.size%in_len==0):
                return arr.reshape((np.int(arr.size/in_len),in_len));
            else:
                return np.resize(arr,(np.int(arr.size/in_len)+1,in_len));
        if(axis==1):
            if(arr.size%in_len==0):
                return arr.reshape(in_len,(np.int(arr.size/in_len)));
            else:
                return np.resize(arr,(in_len,np.int(arr.size/in_len)+1));
    def createData(self):
        self.d = np.zeros((2**self.k,self.k))
        j=0
        #find all possible binary numbers of length k
        from itertools import product
        for i in product([0,1], repeat=self.k):
            self.d[j,:] = i;
            j=j+1; 
        self.cwords=np.zeros((2**self.k,self.N));
        self.d = np.int32(self.d);#make d integer
       # for p in range(0,2**k):
       #    cwords[p,:] = Encoder(d[p,:]) #find all possible codewords
        self.cwords = self.Encoder(self.d);
    def createPermutationsOf(self,words,num_chosen=200):
        num_testwords = np.shape(words)[0]
        #word_length = np.shape(words)[1]
        test_choices = np.random.randint(0,num_testwords,num_chosen)
        chosen = words[test_choices[0]]
        for i in range(1,num_chosen):
            chosen = np.append(chosen,words[test_choices[i]]);
         
        return chosen;    
    #Convert recieved bits to message
    def text_from_bits(self,bits):
        n = int(''.join(map(str, bits)), 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
    #Convert message string to bits
    def text_to_bits(self,text):
        bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
        return np.asarray(list(map(int, bits.zfill(8 * ((len(bits) + 7) // 8)))),dtype=int)
    def sendStringMessage(self,msg_str,sendEbN0=10):
        #my_str = np.loadtxt("Input_150_039.txt")
        #send_msg_str = "Big Test this to work right now blah";
        send_msg_str = msg_str
        self.EbN0 = sendEbN0;
        msg_bits = self.text_to_bits(send_msg_str);
        rec_msg_bits = self.sendThroughCommSystem(msg_bits);
        str_ber = self.BER(msg_bits,rec_msg_bits);
        rec_msg_bits = np.int32(rec_msg_bits)
        print(f'Total Message Bits = {msg_bits.size}')
        print (f'Bit Errors = {str_ber*msg_bits.size}\n')
        
        rec_msg_str = self.text_from_bits(rec_msg_bits);
        
        
        print("Transmitted:")
        print(send_msg_str)  
        print("\nRecieved:")
        print(f'{rec_msg_str}\n')  