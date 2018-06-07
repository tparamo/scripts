from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import Callback
from keras.models import Sequential

losses = []

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.handleLoss(logs.get('loss'))

    def handleLoss(self, loss):
        global losses
        losses += [loss]


class doNN:

    def __init__(self, timesteps, dim, samples):
        self.dim = dim
        self.timesteps = timesteps
        self.samples = samples

        # design network model = Sequential()
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))
        self.model.add(LSTM(50, input_shape=(self.timesteps, self.dim), return_sequences=True))

        self.model.add(Dense(self.dim))
        self.model.compile(loss='mae', optimizer='adam')

    def train(self, data):
        data.shape = (int(self.samples / self.timesteps), self.timesteps, self.dim)
        self.model.fit(data, data, epochs=50, batch_size=100, validation_data=(data, data), verbose=0, shuffle=False,
                       callbacks=[LossHistory()])
        data.shape = (self.samples, self.dim)

    def score(self, data):
        data.shape = (int(self.samples / self.timesteps), self.timesteps, self.dim)
        yhat = self.model.predict(data)
        yhat.shape = (self.samples, self.dim)
        return yhat