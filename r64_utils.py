import pandas as pd
import binascii
import struct
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA as sklearnPCA

def readRobotAxisData(filename, suffix=""):
    # Read headers
    headers = []
    data_set = filename + '.dat'
    with open(data_set, 'r') as f:
        lines = f.readlines();
        for l in range(0, len(lines)):
            if lines[l].__contains__('#BEGINCHANNELHEADER') and lines[l + 4].__contains__('EXPLIZIT'):
                header = lines[l + 1].split(',')[-1:][0].rstrip() + suffix
                headers.append(header.strip())

    # Read data
    values = []
    data_set = filename + ".r64"
    with open(data_set, 'rb') as f:
        cont = 0
        for block in iter(lambda: f.read(8), b''):
            hex = binascii.hexlify(block).decode("UTF-8")
            reverse = "".join(reversed([hex[i:i + 2] for i in range(0, len(hex), 2)]))
            decimal = struct.unpack("d", struct.pack("Q", int("0x" + reverse, 16)))[0]
            values.append(decimal)

    return headers, values

def prepareDataFrame(filename, suffix=""):
    headers, values = readRobotAxisData(filename, suffix)
    rows = [values[i:i + len(headers)] for i in range(0, len(values), len(headers))]
    df = pd.DataFrame(rows, columns=headers)
    return df

def prepareNomalisedData(filename, naxis=6):
    concat_df = concatData(filename, naxis=6)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_norm = scaler.fit_transform(concat_df.values)
    return data_norm, concat_df

def prepareSequencialReducedData(filename, naxis=6):
    data_norm, concat_df = prepareNomalisedData(filename, naxis=6)

    sklearn_pca = sklearnPCA(n_components=5)
    XPCA = sklearn_pca.fit_transform(data_norm)
    print(sklearn_pca.explained_variance_ratio_)

    aux = pd.to_datetime(pd.date_range('1/1/2018 00:00:00.006392', periods=len(data_norm), freq='4ms')).values
    df = pd.DataFrame(XPCA, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'])
    df.insert(len(df.columns), 'time', aux)
    return df

def concatData(filename, naxis=6):
    data_frames = []
    new_columns = []

    for i in range(0, naxis):
        data_set = filename + "#" + str(i + 1)
        suffix = "_" + str(i + 1)
        df = prepareDataFrame(data_set, suffix)
        data_frames.append(df)

    concat_df = pd.concat(data_frames, axis=1)
    return concat_df

def prepareSequencialData(filename, naxis=6):
    df = concatData(filename, naxis=6)
    aux = pd.to_datetime(pd.date_range('1/1/2018 00:00:00.006392', periods=len(df), freq='4ms')).values[]
    df.insert(len(df.columns), 'time', aux)
    return df