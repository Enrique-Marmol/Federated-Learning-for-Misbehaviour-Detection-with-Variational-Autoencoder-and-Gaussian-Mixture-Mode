import math
import os
import time
from typing import Optional, Tuple, Dict

import numpy

import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
from sklearn.mixture import GaussianMixture
import numpy as np
from Autoencoder import *

learn = 0.005
epochs = 1
alpha = 1

nc = 298
cvt = "diag"

def set_epochs(ep):
    glo = globals()
    glo['epochs'] = ep


def set_alpha(al):
    glo = globals()
    glo['alpha'] = al


def set_lr(lr):
    glo = globals()
    glo['learn'] = lr


def load_data(path):
    # load dataset
    training_dataset = pd.read_csv(path)
    # testing_dataset = pd.read_csv("/home/enrique/IBMFL1.0.4/testings/testing_semicomun_vehicles.csv", delimiter=",")

    # pre-process the data
    training_dataset = preprocess(training_dataset)
    # testing_dataset = preprocess(testing_dataset)

    # split the data
    x_0 = training_dataset.iloc[:, :-1]
    y_0 = training_dataset.iloc[:, -1]

    x = np.array(x_0)
    y = np.array(y_0)
    """
    x_t_0 = testing_dataset.iloc[:, :-1]
    y_t_0 = testing_dataset.iloc[:, -1]

    x_train = np.array(x_0)
    sy_train = np.array(y_0)

    self.x_test = np.array(x_t_0)
    self.y_test = np.array(y_t_0)
    """
    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)  # , random_state=42)
    """
    x_test = np.concatenate((self.x_test, x_t_0), axis=0)
    y_test = np.concatenate((self.y_test, y_t_0), axis=0)
    """
    return (x_train, y_tr), (x_test, y_te)


def preprocess(training_data):
    # Transform INF and NaN values to median
    pd.set_option('use_inf_as_na', True)
    training_data.fillna(training_data.median(), inplace=True)

    # Shuffle samples
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    # Normalize values
    scaler = MinMaxScaler()

    features_to_normalize = training_data.columns.difference(['Label'])

    training_data[features_to_normalize] = scaler.fit_transform(training_data[features_to_normalize])

    # Return preprocessed data
    return training_data

def leer_cliente_veremi(client_n):
    dir_base = os.getcwd()
    #dir_tam = "Coches_300-275.csv"
    dir_tam = "vehiculos_298.csv"
    #dir_tam = "vehiculos_"+str(nc)+".csv"
    dir_lista_tam = os.path.join(dir_base, dir_tam)
    dir_lista_tam = os.path.abspath(dir_lista_tam)

    lista_clientes = pd.read_csv(dir_lista_tam)
    lista_clientes = np.array(lista_clientes['Vehicle'])
    lc_ind = client_n - 1
    dir_datos = "SMOTETomek/data_party" + str(int(lista_clientes[lc_ind])) + ".csv"
    # dir_datos = "originales/data_party" + str(lista_clientes[lc_ind]) + ".csv"
    dir_lista_datos = os.path.join(dir_base, dir_datos)
    dir_lista_datos = os.path.abspath(dir_lista_datos)
    return dir_lista_datos



def create_histogram_veremi(Normal, A, n_clusters, cv_type):
    useless = ['Label']


    Normal.drop(useless, axis=1, inplace=True)
    A.drop(useless, axis=1, inplace=True)

    Normal_ = pd.DataFrame(Normal.iloc[:, :], columns=Normal.columns[:])
    A_ = pd.DataFrame(A.iloc[:, :], columns=A.columns[:])

    Normal_.sample(frac=1).reset_index(drop=True)
    A_.sample(frac=1).reset_index(drop=True)

    nl = len(Normal_)
    nl = int(0.8 * nl)
    Normal_test = Normal_.iloc[nl:, :].reset_index(drop=True)
    A_test = A_.reset_index(drop=True)

    Normal_test["target"] = 0
    A_test["target"] = 1

    Normal_train = Normal_.iloc[:nl, :].reset_index(drop=True)

    ini = time.time()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=cv_type, random_state=0)
    gmm.fit(Normal_train)
    fin = time.time()
    print("Tiempo crear y entrenar GMM: " + str(fin-ini))
    # print(gmm.means_)
    desv = Normal_train.std()
    centers = gmm.means_

    # y se hace el predict del cluster
    clus = pd.Series(gmm.predict(Normal_train), name='Clusters')
    cluster_final = pd.concat([Normal_train, clus], axis=1)

    n_inputs_mud = Normal_train.shape[1]

    return Normal_test, A_test, Histo_numpy_2(cluster_final, centers, desv, n_inputs_mud, n_clusters,
                                              1), desv, centers, gmm

def Histo_numpy_2(datos, centros, desviaciones, n_variables, clusters, W):
    datos_aux = datos
    col = datos.columns
    datos_aux = datos_aux.drop([col[-1]], axis=1)
    datos_aux = np.asarray(datos_aux)
    centros = np.asarray(centros)
    desviaciones = np.asarray(desviaciones)
    centros = centros[None, :, :]    # (   1, 250, 17)
    datos = datos_aux[:, None, :]        # (1000,   1, 17)
    desv = W * desviaciones
    Weight = 1 / n_variables
    sup = centros + desv
    inf = centros - desv
    H = Weight * ((datos < sup) & (datos > inf)).sum(axis=2)
    return H


def create_autoencoder_rbm(hist, layers):
    layer_dim = []
    for i in range(len(layers)): layer_dim.append(layers[i])
    x = np.transpose(hist)
    autoencoder = Autoencoder(layer_dims=layer_dim)
    autoencoder.pretrain(x, epochs=15, num_samples=1000)
    autoencoder.save("pretrained_weights")
    model = autoencoder.unroll()
    opt = RMSprop(learning_rate=learn)
    #model.compile(optimizer='rmsprop', loss='mse')
    model.compile(optimizer=opt, loss='mse')
    return model

def Deteccion(datos, umbral, variables, clusters, NN, centers, desv, gmm, client):
    #print("calculando detencion")
    probs = pd.DataFrame(np.exp(gmm.score_samples(datos.iloc[:, 0:variables])), columns=["Probability"])
    probs["Prediction"] = 'D'
    probs.loc[probs.Probability == 0, "Prediction"] = 'A'
    probs.loc[probs.Probability >= 1, "Prediction"] = 'N'
    probs.loc[probs.Prediction == 'A', "Prediction"] = 1.0
    probs.loc[probs.Prediction == 'N', "Prediction"] = 0.0
    probs.loc[probs.Prediction == 'D', "Prediction"] = 0.5
    longitud_total = len(probs)

    #His_t = Histo(datos, centers, desv, variables, clusters, 1)
    His_t = Histo_numpy_2(datos, centers, desv, variables, clusters, 1)
    dif = NN.predict(His_t) - His_t
    REC_t = np.sqrt(np.sum((dif) ** 2, axis=1))
    sug = []
    #print(REC_t)

    for i in range(len(REC_t)):
        if (REC_t[i] < umbral):
            sug.append(0)
        else:
            sug.append(1)

    probs["Sugerencia"] = sug
    probs.loc[probs.Prediction == 0.5, 'Prediction'] = probs.loc[probs.Prediction == 0.5, 'Sugerencia']
    probs["Target"] = datos["target"].reset_index(drop=True)
    jeje=probs[probs.Probability < 1]
    jeje=jeje[jeje.Probability>0]
    #print("Longitud AE party "+str(client)+": " + str(len(jeje)))
    proporcion_ae = len(jeje)/longitud_total
    jeje.to_csv(os.getcwd() + "/acc_autoencoder.csv")
    M = jeje[["Prediction", "Target"]]
    TP = len(M[(M.Prediction == 1) & (M.Target == 1)])
    FN = len(M[(M.Prediction == 0) & (M.Target == 1)])
    FP = len(M[(M.Prediction == 1) & (M.Target == 0)])
    TN = len(M[(M.Prediction == 0) & (M.Target == 0)])
    acc_ae = (TP + TN) / (TP + TN + FP + FN)
    #print("ACCURACY AUTOENCODER: " + str(acc))
    aux = []
    path = os.getcwd()+"/metricas_autoencoder/acc_ae_client_"+str(client)+".csv"
    col_name = ['Accuracy']
    lista = {
        "Accuracy": acc_ae
    }
    aux.append(lista)
    df1 = pd.DataFrame(aux, columns=col_name)
    df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))


    gmm = probs[probs.Probability >= 1]
    gmm2 = probs[probs.Probability == 0]
    #print("Longitud GMM party"+str(client)+": " + str(len(gmm) + len(gmm2)))
    proporcion_gmm = (len(gmm)+len(gmm2))/longitud_total
    M = gmm[["Prediction", "Target"]]
    TP_1 = len(M[(M.Prediction == 1) & (M.Target == 1)])
    FN_1 = len(M[(M.Prediction == 0) & (M.Target == 1)])
    FP_1 = len(M[(M.Prediction == 1) & (M.Target == 0)])
    TN_1 = len(M[(M.Prediction == 0) & (M.Target == 0)])

    M = gmm2[["Prediction", "Target"]]
    TP_2 = len(M[(M.Prediction == 1) & (M.Target == 1)])
    FN_2 = len(M[(M.Prediction == 0) & (M.Target == 1)])
    FP_2 = len(M[(M.Prediction == 1) & (M.Target == 0)])
    TN_2 = len(M[(M.Prediction == 0) & (M.Target == 0)])


    TP = TP_1 + TP_2
    FN = FN_1 + FN_2
    FP = FP_1 + FP_2
    TN = TN_1 + TN_2

    acc_gmm = (TP + TN) / (TP + TN + FP + FN)

    #print("ACCURACY GMM: " + str(acc))
    aux = []
    path = os.getcwd() + "/metricas_autoencoder/acc_gmm_client_" + str(client) + ".csv"
    col_name = ['Accuracy']
    lista = {
        "Accuracy": acc_gmm
    }
    aux.append(lista)
    df1 = pd.DataFrame(aux, columns=col_name)
    df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))



    return probs, acc_ae, proporcion_ae, acc_gmm, proporcion_gmm

def my_metrics(Normal_test, A_test, variables, UM, n, model, centers, desv, gmm, clusters, client):
    #print("calculando metricas")
    rs = random.randint(0, 42)
    split = pd.concat([Normal_test.sample(n, random_state=rs).reset_index(drop=True),
                       A_test.sample(n, random_state=rs).reset_index(drop=True)])

    # FINAL = Deteccion(split, UM, 36, 3, model, centers, desv, gmm)

    FINAL, acc_ae, p_ae, acc_gmm, p_gmm = Deteccion(split, UM, variables, clusters, model, centers, desv, gmm,client)
    M = FINAL[["Prediction", "Target"]]

    TP = len(M[(M.Prediction == 1) & (M.Target == 1)])
    FN = len(M[(M.Prediction == 0) & (M.Target == 1)])
    FP = len(M[(M.Prediction == 1) & (M.Target == 0)])
    TN = len(M[(M.Prediction == 0) & (M.Target == 0)])

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP / (TP + FP))
    recall = TP / (TP + FN)
    f1_sc = 2 * (recall * precision) / (recall + precision)

    mcc = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    return acc, precision, recall, f1_sc, mcc, acc_ae, p_ae, acc_gmm, p_gmm

class tfmlpClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, centers, desv, gmm, n_component, n_inputs_mud, directorio_metricas,
                 party_number):
        self.model = model
        self.x_train = x_train
        self.y_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.centers = centers
        self.desv = desv
        self.gmm = gmm
        self.n_component = n_component
        self.n_inputs_mud = n_inputs_mud
        self.dir_met = directorio_metricas
        self.party_number = party_number

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        # self.model.set_weights(parameters)
        mean_weights = parameters

        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        rnd = config["round"]
        steps = config["val_steps"]
        rondas = config["rounds"]
        """
        if rnd == 1:
            self.model.set_weights(parameters)
        """
        print("Ronda: " + str(rnd))
        # Train the model using hyperparameters from config
        if rnd > 1:
            for epoch in range(epochs):
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    #batch_size,
                    epochs = 1,
                    validation_split=0.1,
                )
                theta = 1/(1+(learn*alpha))
                #theta = alpha
                new_param = fedplus(self.model.get_weights(), mean_weights, theta)
                self.model.set_weights(new_param)
        else:
            print("primera ronda fit normal")
            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train,
                self.y_train,
                #batch_size,
                epochs = epochs,
                validation_split=0.1,
            )
            new_param = self.model.get_weights()

        parameters_prime = new_param
        num_examples_train = len(self.x_train)


        Hpred = pd.DataFrame(self.model.predict(self.x_train))
        REC = np.sqrt(np.sum((self.x_train - Hpred) ** 2, axis=1))
        #UM = np.mean(REC) + 0.01 * np.std(REC)
        UM = np.quantile(REC, 0.95)
        n = int(len(self.x_test))

        accuracy, recall, precision, f1_sc, mcc, acc_ae, p_ae, acc_gmm, p_gmm = my_metrics(self.x_test, self.y_test, self.n_inputs_mud, UM, n, self.model,
                                                             self.centers,
                                                             self.desv, self.gmm, self.n_component, self.party_number)

        results = {
            "accuracy": accuracy,
            "loss": history.history["loss"][-1],
            "val_loss": 0,
            "val_accuracy": accuracy,
        }

        # Guardar en csv externo
        aux = []
        path = self.dir_met
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Loss', 'Accuracy_ae', 'Proporcion_ae', 'Accuracy_gmm', 'Proporcion_gmm']
        lista = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1_score": f1_sc,
            "Matthew_Correlation_coefficient": mcc,
            "Loss": history.history["loss"][-1],
            "Accuracy_ae": acc_ae,
            "Proporcion_ae": p_ae,
            "Accuracy_gmm": acc_gmm,
            "Proporcion_gmm": p_gmm
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        # loss, accuracy, *aux = self.model.evaluate(self.x_train, self.y_train, 32, steps=steps)
        Hpred = pd.DataFrame(self.model.predict(self.x_train))
        REC = np.sqrt(np.sum((self.x_train - Hpred) ** 2, axis=1))
        UM = np.mean(REC) + 0.01 * np.std(REC)
        # UM = 0.0005
        n = int(len(self.x_test))

        accuracy, recall, precision, f1_sc, mcc = my_metrics(self.x_test, self.y_test, self.n_inputs_mud, UM, n, self.model,
                                                             self.centers,
                                                             self.desv, self.gmm, self.n_component, self.party_number)

        loss = np.mean(REC)
        """
        print("loss: " + str(loss))
        print("accuracy: " + str(accuracy))
        print("recall: " + str(recall))
        print("precision: " + str(precision))
        print("f1_sc: " + str(f1_sc))
        print("mcc: " + str(mcc))
        """
        num_examples_test = len(self.x_test)

        # Guardar en csv externo
        aux = []
        path = self.dir_met
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Loss']
        lista = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1_score": f1_sc,
            "Matthew_Correlation_coefficient": mcc,
            "Loss": loss
        }
        """
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        """
        return loss, num_examples_test, {"accuracy": accuracy}


def fedplus(weights, mean, theta):
    z = numpy.asarray(mean)
    weights = numpy.asarray(weights)

    fedp = theta * weights + (1 - theta) * z
    return fedp


def start_client(client_n):
    dir_base = os.getcwd()
    cliente = leer_cliente_veremi(client_n)
    print(cliente)
    dataset = pd.read_csv(cliente)

    scaler = MinMaxScaler()
    features_to_normalize = dataset.columns.difference(['Label'])
    dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

    pd.set_option('use_inf_as_na', True)
    dataset.fillna(dataset.median(), inplace=True)
    dataset = dataset.sample(frac=1).reset_index(drop=True)


    # Limpiamos el csv
    car_metricas = "metricas_parties"
    try:
        os.mkdir(car_metricas)
    except FileExistsError:
        aux = 0

    dir_metricas = 'metricas_parties/history_' + str(client_n) + '.csv'
    dir_metricas_total = os.path.join(dir_base, dir_metricas)
    dir_metricas_total = os.path.abspath(dir_metricas_total)

    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient','Loss',
                'Accuracy_ae', 'Proporcion_ae', 'Accuracy_gmm', 'Proporcion_gmm']#,'Cohen_Kappa_Score']
    path = dir_metricas_total
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    # Load a subset of CIFAR-10 to simulate the local data partition

    Normal = dataset[dataset.Label == 0]
    A = dataset[dataset.Label > 0]

    nl = len(Normal)
    nl = int(0.8 * nl)
    nl = len(Normal) - nl
    print(nl)

    at = int(nl / 5)
    resto = nl - 5 * at
    """
    at_1 = A[A.Label == 1]
    at_2 = A[A.Label == 2]
    at_3 = A[A.Label == 3]
    at_4 = A[A.Label == 4]
    at_5 = A[A.Label == 5]

    at_1 = at_1[0:at]
    at_2 = at_2[0:at]
    at_3 = at_3[0:at + resto]
    at_4 = at_4[0:at]
    at_5 = at_5[0:at]
    A = pd.concat([at_1, at_2, at_3, at_4, at_5], ignore_index=True)
    """
    n_component = 298#nc  # 244
    cv_type = cvt
    ini = time.time()
    Normal_t, A_t, hist, desv, centers, gmm = create_histogram_veremi(Normal, A, n_component, cv_type)
    n_inputs = hist.shape[1]
    #mitad_layers = [n_inputs, 200, 100]
    mitad_layers = [n_inputs, int(n_inputs/2), int(n_inputs/3)]
    model = create_autoencoder_rbm(hist, mitad_layers)
    print(learn)
    n_inputs_mud = Normal.shape[1]

    # Start Flower client
    client = tfmlpClient(model, hist, hist, Normal_t, A_t, centers, desv, gmm, n_component, n_inputs_mud, dir_metricas_total,
                         client_n)
    print(("party " + str(client_n) + " lista"))
    fin = time.time()
    print("TIEMPO party "+str(client_n)+": "+str(fin-ini))
    # IP lorien 155.54.95.95
    fl.client.start_numpy_client("[::]:8080", client=client)


def start_client_vae(client_n, lr):
    dir_base = os.getcwd()
    cliente = leer_cliente_veremi(client_n)
    print(cliente)
    dataset = pd.read_csv(cliente)

    scaler = MinMaxScaler()
    features_to_normalize = dataset.columns.difference(['Label'])
    dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

    pd.set_option('use_inf_as_na', True)
    dataset.fillna(dataset.median(), inplace=True)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Limpiamos el csv
    car_metricas = "metricas_parties"
    try:
        os.mkdir(car_metricas)
    except FileExistsError:
        aux = 0

    dir_metricas = 'metricas_parties/history_' + str(client_n) + '.csv'
    dir_metricas_total = os.path.join(dir_base, dir_metricas)
    dir_metricas_total = os.path.abspath(dir_metricas_total)

    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient', 'Loss',
                'Accuracy_ae', 'Proporcion_ae', 'Accuracy_gmm', 'Proporcion_gmm']  # ,'Cohen_Kappa_Score']
    path = dir_metricas_total
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    # Load a subset of CIFAR-10 to simulate the local data partition

    Normal = dataset[dataset.Label == 0]
    A = dataset[dataset.Label > 0]
    """
    nl = len(Normal)
    nl = int(0.8 * nl)
    nl = len(Normal) - nl
    print(nl)

    at = int(nl / 5)
    resto = nl - 5 * at

    at_1 = A[A.Label == 1]
    at_2 = A[A.Label == 2]
    at_3 = A[A.Label == 3]
    at_4 = A[A.Label == 4]
    at_5 = A[A.Label == 5]

    at_1 = at_1[0:at]
    at_2 = at_2[0:at]
    at_3 = at_3[0:at + resto]
    at_4 = at_4[0:at]
    at_5 = at_5[0:at]
    A = pd.concat([at_1, at_2, at_3, at_4, at_5], ignore_index=True)
    """
    n_component = nc  # 244
    cv_type = cvt
    Normal_t, A_t, hist, desv, centers, gmm = create_histogram_veremi(Normal, A, n_component, cv_type)
    n_inputs = hist.shape[1]
    n_inputs_mud = Normal.shape[1]

    def sample(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    """
    def vae_loss(x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + 0.01 * kl_loss)
        return total_loss
    """
    original_dim = hist.shape[1]
    input_shape = (original_dim,)
    intermediate_dim = int(original_dim / 2)
    latent_dim = int(original_dim / 3)

    # encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use the reparameterization trick and get the output from the sample() function
    z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, z, name='encoder')
    #encoder.summary()

    # decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    # Instantiate the decoder model:
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    # full VAE model
    outputs = decoder(encoder(inputs))
    vae_model = Model(inputs, outputs, name='vae_mlp')


    # the KL loss function:
    def vae_loss(x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + 0.001 * kl_loss)
        # total_loss = reconstruction_loss + kl_loss
        return total_loss

    from tensorflow import optimizers
    # opt = optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    set_lr(lr)
    opt = optimizers.RMSprop(learning_rate=learn)
    vae_model.compile(optimizer=opt, loss=vae_loss)
    print(learn)

    #model = vae_model


    # Start Flower client
    client = tfmlpClient(vae_model, hist, hist, Normal_t, A_t, centers, desv, gmm, n_component, n_inputs_mud,
                         dir_metricas_total,
                         client_n)
    print(("party " + str(client_n) + " lista"))

    # IP lorien 155.54.95.95
    fl.client.start_numpy_client("[::]:8080", client=client)


def start_server(parties, rounds, epoch, alph, lr):
    dir_base = os.getcwd()
    dataset = leer_cliente_veremi(1)
    dataset = pd.read_csv(dataset)
    scaler = MinMaxScaler()
    features_to_normalize = dataset.columns.difference(['Label'])
    dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

    Normal = dataset[dataset.Label == 0]
    A = dataset[dataset.Label > 0]

    nl = len(Normal)
    nl = int(0.8 * nl)
    nl = len(Normal) - nl
    print(nl)

    at = int(nl / 5)
    resto = nl - 5 * at

    at_1 = A[A.Label == 1]
    at_2 = A[A.Label == 2]
    at_3 = A[A.Label == 3]
    at_4 = A[A.Label == 4]
    at_5 = A[A.Label == 5]

    at_1 = at_1[0:at]
    at_2 = at_2[0:at]
    at_3 = at_3[0:at + resto]
    at_4 = at_4[0:at]
    at_5 = at_5[0:at]
    A = pd.concat([at_1, at_2, at_3, at_4, at_5], ignore_index=True)

    n_component = 298#nc  # 244
    cv_type = cvt
    Normal_t, A_t, hist, desv, centers, gmm = create_histogram_veremi(Normal, A, n_component, cv_type)
    n_inputs = hist.shape[1]
    mitad_layers = [n_inputs, int(n_inputs/2), int(n_inputs/3)]
    model = create_autoencoder_rbm(hist, mitad_layers)


    set_epochs(epoch)
    set_alpha(alph)
    set_lr(lr)
    theta = 1 / (1 + lr * alph)
    print("THETA: "+ str(theta))

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=parties,
        min_eval_clients=parties,
        min_available_clients=parties,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": rounds, "local_epochs": epoch}, strategy=strategy)


def start_server_vae(parties, rounds, epoch, alph, lr):
    dir_base = os.getcwd()
    dataset = leer_cliente_veremi(1)
    dataset = pd.read_csv(dataset)
    scaler = MinMaxScaler()
    features_to_normalize = dataset.columns.difference(['Label'])
    dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

    Normal = dataset[dataset.Label == 0]
    A = dataset[dataset.Label > 0]
    """
    nl = len(Normal)
    nl = int(0.8 * nl)
    nl = len(Normal) - nl
    print(nl)

    at = int(nl / 5)
    resto = nl - 5 * at

    at_1 = A[A.Label == 1]
    at_2 = A[A.Label == 2]
    at_3 = A[A.Label == 3]
    at_4 = A[A.Label == 4]
    at_5 = A[A.Label == 5]

    at_1 = at_1[0:at]
    at_2 = at_2[0:at]
    at_3 = at_3[0:at + resto]
    at_4 = at_4[0:at]
    at_5 = at_5[0:at]
    A = pd.concat([at_1, at_2, at_3, at_4, at_5], ignore_index=True)
    """
    n_component = nc  # 244
    cv_type = cvt
    Normal_t, A_t, hist, desv, centers, gmm = create_histogram_veremi(Normal, A, n_component, cv_type)
    n_inputs = hist.shape[1]

    def get_error_term(v1, v2, _rmse=True):
        if _rmse:
            return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
        # return MAE
        return np.mean(abs(v1 - v2), axis=1)

    def sample(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    original_dim = hist.shape[1]
    input_shape = (original_dim,)
    intermediate_dim = int(original_dim / 2)
    latent_dim = int(original_dim / 3)

    # encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use the reparameterization trick and get the output from the sample() function
    z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, z, name='encoder')
    # encoder.summary()

    # decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    # Instantiate the decoder model:
    decoder = Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()

    # full VAE model
    outputs = decoder(encoder(inputs))
    vae_model = Model(inputs, outputs, name='vae_mlp')

    # the KL loss function:
    def vae_loss(x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + 0.01 * kl_loss)
        # total_loss = reconstruction_loss + kl_loss
        return total_loss

    from tensorflow import optimizers

    opt = optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    # opt = optimizers.RMSprop(learning_rate=0.005)

    vae_model.compile(optimizer=opt, loss=vae_loss)

    model = vae_model

    set_epochs(epoch)
    set_alpha(alph)
    set_lr(lr)
    theta = 1 / (1 + lr * alph)
    print("THETA: "+ str(theta))

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=parties,
        min_eval_clients=parties,
        min_available_clients=parties,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": rounds, "local_epochs": epoch}, strategy=strategy)



def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # The `evaluate` function will be called after every round
    def evaluate(
            weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)

        # Update model with the latest parameters
        # loss, accuracy = model.evaluate(hist, hist)
        loss = 0
        accuracy = 0

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": epochs,  # if rnd < 2 else 2,
        "round": rnd,
        "val_steps": 5,
        "rounds": 10
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    # return MAE
    return np.mean(abs(v1 - v2), axis=1)









