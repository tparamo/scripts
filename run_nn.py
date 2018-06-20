from keras.models import model_from_yaml
import LSTM
import r64_utils as r64
import pandas as pd


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
    df.shape = (int(22500 / 10), 10, 2)
    nn = LSTM(10, 2, 22500)
    for i in range(20):
        out = "Iteration " + str(i)
        print(out)
        nn.train(df)
        yhat = nn.score(df)
        df.shape = (22500, 2)

def runDataSets():
    base_dir = "/home/tere/Data/Edincubator/datos/dataset_1/"
    suffix = "_NextGenDrive"
    robots = ["KBU1A1121650R02", "KABHVS111110R01", "KASTAL123860R01"]
    for robot in robots:
        df = r64.prepareSequencialData(base_dir+robot+suffix)
        for column in df.columns[::-1]:
            if column != "time":
                invidual_df = pd.concat([df['time'], df[column]], axis=1)
                invidual_df.columns = ['time', 'VAR']
                run(invidual_df, base_dir+robot, column)


if __name__=='__main__':
    trainModel(df)
