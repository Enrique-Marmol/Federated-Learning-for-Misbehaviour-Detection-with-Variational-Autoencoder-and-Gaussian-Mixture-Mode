from FuncionesAux import *
import warnings
import os
import argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

warnings.filterwarnings("ignore")
import multiprocessing

warnings.filterwarnings("ignore")
import time

parametros = argparse.ArgumentParser()
parametros.add_argument("Clients", help="Number of clients", type=int)
parametros.add_argument("Rounds", help="Number of rounds", type=int)
parametros.add_argument("Epochs", help="Number of epochs", type=int)
parametros.add_argument("Alpha", help="Alpha", type=float)
parametros.add_argument("Lr", help="Learning rate", type=float)
param = parametros.parse_args()
n_clientes = param.Clients
rondas = param.Rounds
epocas = param.Epochs
alpha = param.Alpha
lr = param.Lr

acc_media_gb = []
clients = []

server = multiprocessing.Process(target=start_server_vae, args=(n_clientes, rondas, epocas, alpha, lr))
server.start()
time.sleep(30)

for i in range(1,n_clientes+1):
    p = multiprocessing.Process(target=start_client_vae, args=(i,lr))
    p.start()
    clients.append(p)

server.join()
for client in clients:
    client.join()
fin = time.time()


time.sleep(5)
