import numpy as np

class OFDM_module():
    def __init__(self, data,EbN0=5):
        self.OFDM_data = data
        self.K = 7;
        self.CP = 5
        self.allCarriers = np.arange(self.K)
        self.channelResponse = np.array([1, 0, 0.3+0.3j,0.2-0.5j,0, 0.1j])
        #self.channelResponse = np.array([1]);#change!
        self.SNRdb = EbN0
        #self.SNRdb = 100;
    def IDFT(self, OFDM_data):
        return np.fft.ifft(OFDM_data)

    def addCP(self, OFDM_time):
        cp = OFDM_time[-self.CP:]
        return np.hstack([cp, OFDM_time])

    def channel(self, signal):
        convolved = np.convolve(signal, self.channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-self.SNRdb/10)
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return convolved + noise

    def removeCP(self, signal):
        return signal[self.CP:(self.CP+self.K)]

    def DFT(self, OFDM_RX):
        return np.fft.fft(OFDM_RX)

    def OFDM_run(self):
        OFDM_time = self.IDFT(self.OFDM_data)
        OFDM_withCP = self.addCP(OFDM_time)
        OFDM_TX = OFDM_withCP
        OFDM_RX = self.channel(OFDM_TX)
        OFDM_RX_noCP = self.removeCP(OFDM_RX)
        OFDM_demod = self.DFT(OFDM_RX_noCP)
        return OFDM_demod
    def setSNR(self,EbN0):
        self.SNRdb = EbN0;
    def getSNR(self):
        return self.SNRdb;
