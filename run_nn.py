from keras.models import model_from_yaml
import LSTM
from ml


def saveModel(nn, filename):
    # serialize model to YAML
    model_yaml = nn.model.to_yaml()
    with open(filename+ '.yaml', "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    nn.model.save_weights(filename + '_weights.h5')

def loadModel(filename):
    yaml_file = open(filename+ '.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(filename.h5 + '_weights.h5')
    return loaded_model

def trainModel(df):
    XPCA.shape = (int(22500 / 10), 10, 2)
    nn = LSTM(10, 2, 22500)
    for i in range(20):
        out = "Iteration " + str(i)
        print(out)
        nn.train(XPCA)
        yhat = nn.score(XPCA)
        XPCA.shape = (22500, 2)

if __name__=='__main__':
    trainModel(df)
