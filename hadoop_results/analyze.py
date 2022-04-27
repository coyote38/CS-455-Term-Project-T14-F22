import os 
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from collections import deque
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


nn_params = {
    "N_STEPS": 4,
    "FEATURE_COLUMNS": ["date", "power_gen"],
    "LAYERS": [(128, LSTM), (128, Dense), (128, Dense)],
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "DROPOUT": .4,
    "EPOCHS": 1000,
    "BATCH_SIZE": 8,
    "PATIENCE": 100,
    "LOOKUP_STEP": 1
}

def layer_name_converter(layer):
    # print(layer, flush=True)
    string = ""
    
    if str(layer[1]) == "<class 'keras.layers.recurrent_v2.LSTM'>":
        string += "LSTM"
    elif str(layer[1]) == "<class 'keras.layers.recurrent.SimpleRNN'>":
        string += "SRNN"
    elif str(layer[1]) == "<class 'keras.layers.recurrent_v2.GRU'>":
        string += "GRU"
    elif str(layer[1]) == "<class 'keras.layers.core.dense.Dense'>":
        string += "Dense"
    elif str(layer[1]) == "<class 'tensorflow_addons.layers.esn.ESN'>":
        string += "ESN"
    else:
        string += str(layer[1])

    return string

def layers_string(layers):
    string = "["
    for layer in layers:
        string += "(" + str(layer[0]) + "-" + layer_name_converter(layer) + ")"
    string += "]"

    return string

def get_model_name(nn_params):
    return (f"""{nn_params["FEATURE_COLUMNS"]}-layers{layers_string(nn_params["LAYERS"])}-step"""
            f"""{nn_params["N_STEPS"]}-epoch{nn_params["EPOCHS"]}""" 
            f"""-pat{nn_params["PATIENCE"]}-batch{nn_params["BATCH_SIZE"]}-drop{nn_params["DROPOUT"]}""")

def create_model(params):
    model = Sequential()
    # print(bi_string)
    for layer in range(len(params["LAYERS"])):
        if layer == 0:
            model_first_layer(model, params["LAYERS"], layer, params["N_STEPS"], params["FEATURE_COLUMNS"])
        elif layer == len(params["LAYERS"]) - 1:
            model_last_layer(model, params["LAYERS"], layer)
        else:
            model_hidden_layers(model, params["LAYERS"], layer)
    
        model.add(Dropout(params["DROPOUT"]))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=params["LOSS"], metrics=["mean_absolute_error"], optimizer=params["OPTIMIZER"])

    return model

def model_first_layer(model, layers, ind, n_steps, features):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu", input_shape=(None, len(features))))
    else:
        if next_layer_name == "Dense":
            model.add(layers[ind][1](layers[ind][0], return_sequences=False, 
                input_shape=(n_steps, len(features))))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=True, 
                input_shape=(None, n_steps)))

    return model

def model_hidden_layers(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        if next_layer_name == "Dense":
            model.add(layers[ind][1](layers[ind][0], return_sequences=False))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=True))

    return model

def model_last_layer(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False))
    
    return model

def make_plot(Xtrain, ytrain, xfit, yfit, region, p_type):
    plot_name = f"plots/{region}{p_type}.png"
    plt.scatter(Xtrain, ytrain)
    plt.plot(xfit, yfit)
    plt.title(f"{region} {p_type}")
    plt.xlabel("Date (Years)")
    plt.ylabel("Power Generation for Month (MW)")
    plt.savefig(plot_name)
    plt.close()

df = pd.read_csv("power_generation.csv", names=["region", "date", "power_gen"])

region_frames = {
    "SE":pd.DataFrame(),
    "SW":pd.DataFrame(),
    "MIDW":pd.DataFrame(),
    "CAL":pd.DataFrame(),
    "NW":pd.DataFrame(),
    "NE":pd.DataFrame(),
    "MIDA":pd.DataFrame(),
    "FLA":pd.DataFrame(),
    "CENT":pd.DataFrame(),
    "TEN":pd.DataFrame(),
    "NY":pd.DataFrame(),
    "CAR":pd.DataFrame(),
    "TEX":pd.DataFrame(),

}


# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

df["date"] = df["date"].str.replace('-', "")
df["date"] = pd.to_numeric(df["date"].str.slice(0, 4)) + pd.to_numeric(df["date"].str.slice(4, 6)) / 12
df["date"] = df["date"] - 2015

predict_list = np.array([7.5, 8, 8.5, 9, 9.5, 10]).reshape(-1, 1)

results_list = []



for region in region_frames:
    print(region)
    region_frames[region] = df[df["region"] == region]
    region_frames[region] = region_frames[region].drop(columns=["region"])

    Xtrain, Xtest, ytrain, ytest = train_test_split(region_frames[region]["date"].values, region_frames[region]["power_gen"].values, 
        test_size=.2, shuffle=True)
    Xtrain = Xtrain.reshape(-1, 1)
    Xtest = Xtest.reshape(-1, 1)


    reg = LinearRegression().fit(Xtrain, ytrain)
    reg_score = reg.score(Xtest, ytest)
    reg_pred = reg.predict(predict_list)
    reg_error = mean_absolute_error(ytest, reg.predict(Xtest))
    print(f"Linear regression score: {reg_score}")
    print(f"Linear regression mae: {reg_error}")
    print(f"Linear regression predictions: {reg_pred}")
    xfit = np.linspace(0, 10, 1000)
    yfit = reg.predict(xfit[:, np.newaxis])
    make_plot(Xtrain, ytrain, xfit, yfit, region, "lin_reg")

    fore = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=1, n_jobs=-1)
    fore.fit(Xtrain, ytrain)
    fore_score = fore.score(Xtest, ytest)
    fore_pred = fore.predict(predict_list)
    fore_error = mean_absolute_error(ytest, fore.predict(Xtest))
    print(f"Random forest score: {fore_score}")
    print(f"Random forest regression mae: {fore_error}")
    print(f"Random forest predictions: {fore_pred}")
    xfit = np.linspace(0, 10, 1000)
    yfit = fore.predict(xfit[:, np.newaxis])
    make_plot(Xtrain, ytrain, xfit, yfit, region, "rand_fore")

    knn = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
    knn.fit(Xtrain, ytrain)
    knn_score = knn.score(Xtest, ytest)
    knn_pred = knn.predict(predict_list)
    knn_error = mean_absolute_error(ytest, knn.predict(Xtest))
    print(f"KNN score: {knn_score}")
    print(f"KNN regression mae: {knn_error}")
    print(f"KNN predictions: {knn_pred}")
    xfit = np.linspace(0, 10, 1000)
    yfit = knn.predict(xfit[:, np.newaxis])
    make_plot(Xtrain, ytrain, xfit, yfit, region, "knn")

    
    tf_df = region_frames[region]
    tf_df["future"] = tf_df["power_gen"].shift(-1)

    last_sequence = np.array(tf_df[nn_params["FEATURE_COLUMNS"]].tail(nn_params["LOOKUP_STEP"]))
    sequence_data = []
    sequences = deque(maxlen=nn_params["N_STEPS"])
    for entry, target in zip(tf_df[nn_params["FEATURE_COLUMNS"]].values, tf_df["future"].values):
        sequences.append(entry)
        if len(sequences) == nn_params["N_STEPS"]:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result = {}
    result["last_sequence"] = last_sequence

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    X = np.delete(X, len(X) - 1, 0)
    y = np.delete(y, len(y) - 1, 0)

    

    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=.2,
      shuffle=False)
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_test"], 
        result["y_test"], test_size=.5, shuffle=False)

    # print("train", result["X_train"]) 
    # print("valid", result["X_valid"])
    # print("test", result["X_test"])
    # print(len(result["X_train"]), len(result["X_valid"]), len(result["X_test"]))

    train = Dataset.from_tensor_slices((result["X_train"], result["y_train"]))
    valid = Dataset.from_tensor_slices((result["X_valid"], result["y_valid"]))
    test = Dataset.from_tensor_slices((result["X_test"], result["y_test"]))

    train = train.batch(nn_params["BATCH_SIZE"])
    valid = valid.batch(nn_params["BATCH_SIZE"])
    test = test.batch(nn_params["BATCH_SIZE"])
    
    train = train.cache()
    valid = valid.cache()
    test = test.cache()

    train = train.prefetch(buffer_size=AUTOTUNE)
    valid = valid.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)


    model_name = (region + "-" + get_model_name(nn_params))    
    model = create_model(nn_params)
    early_stop = EarlyStopping(patience=nn_params["PATIENCE"])
    
    history = model.fit(train,
        batch_size=nn_params["BATCH_SIZE"],
        epochs=nn_params["EPOCHS"],
        verbose=2,
        validation_data=valid,
        callbacks = [early_stop]   
    )



    nn_pred = model.predict(test)
    nn_r2 = r2_score(result["y_test"], nn_pred)
    huber_lost, nn_MAE = model.evaluate(test)

    last_sequence = result["last_sequence"][:nn_params["N_STEPS"]]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    nn_pred = model.predict(last_sequence)[0][0]
    

    results_list.append([region, reg_score, reg_error, reg_pred[0], fore_score, fore_error, fore_pred[0], knn_score, 
        knn_error, knn_pred[0], nn_r2, nn_MAE, nn_pred])
    


result_df = pd.DataFrame(results_list, columns=["Region", "Reg r2", "Reg MAE", "Regression Pred", "Forest r2",
    "Forest MAE", "Forest Pred", "KNN r2", "KNN MAE", "KNN Pred", "NN r2", "NN MAE", "NN Pred"])
print(result_df)

#    Region    Reg r2      Reg MAE  Regression Pred  Forest r2   Forest MAE   Forest Pred    KNN r2      KNN MAE      KNN Pred     NN r2       NN MAE       NN Pred
# 0      SE  0.024834  2973.092842      7440.685900   0.515611  1970.397609  11857.487456  0.526862  2032.344379  11826.669855 -2.601358  4211.833008   6475.548828
# 1      SW -0.001589   305.500398      1646.660665   0.738527   158.114013   1385.263970  0.799120   131.643387   1373.269937 -0.161963   132.064667   1501.669067
# 2    MIDW -0.179259  1336.052223      2118.529634   0.011238  1153.968229   1662.901403  0.039994  1203.807700   2417.271943 -0.052250  1112.211670   2403.468262
# 3     CAL -0.311549   342.420642      4052.451161   0.170190   261.462142   4083.138001  0.325681   220.677605   3930.132601 -1.269127   238.330933   4088.081787
# 4      NW  0.065577   184.255311      1955.935858   0.575418   114.641642   1868.663028  0.522133   116.610293   1996.039019 -0.109382   177.338776   2056.588867
# 5      NE -0.005812  1155.497151     10423.302079  -0.162701  1233.392993  10292.711587 -0.176949  1279.053757  10591.563822 -0.194694   678.763428  11250.391602
# 6    MIDA -0.073407  3587.752704      5685.666105  -1.210196  4326.857995  12390.883903 -0.656202  4031.754646   7075.342348 -0.027321  9541.326172   6139.569336
# 7     FLA  0.165865   295.988074      3033.608703   0.748714   140.848993   2871.472563  0.717641   142.522184   2839.286127 -0.766269   249.222046   2666.144287
# 8    CENT -0.093288  5526.291197      7999.794152   0.176221  4723.128349  10987.345044  0.248807  4677.577925   9001.063541  0.365338  3968.116699   6405.539062
# 9     TEN  0.000687  1663.595466     17693.341829  -0.213289  1656.983423  16711.169299  0.300191  1270.315203  16909.838416 -0.084085  1692.079712  17798.478516
# 10     NY  0.035494   998.966125     14003.293878   0.090422  1045.304644  13064.435771  0.023562  1012.573394  13674.880705 -0.432546   853.596436  14302.582031
# 11    CAR -0.067785   521.158499      4048.568150   0.076274   480.181685   3729.551665  0.295816   423.162232   3996.262119 -0.012578   287.055115   3984.182129
# 12    TEX  0.185351  4257.521895    -19821.353822   0.451868  3287.090219 -21668.612408  0.356553  3749.189497 -21516.426600 -4.207795  6341.468750 -14360.631836

# nn_params = {
#     "N_STEPS": 4,
#     "FEATURE_COLUMNS": ["date", "power_gen"],
#     "LAYERS": [(128, LSTM), (128, Dense), (128, Dense)],
#     "LOSS": "huber_loss",
#     "OPTIMIZER": "adam",
#     "DROPOUT": .4,
#     "EPOCHS": 1000,
#     "BATCH_SIZE": 8,
#     "PATIENCE": 100,
#     "LOOKUP_STEP": 1
# }