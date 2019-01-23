#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:55:19 2019

@author: gsus
"""

# Deep Convolutional GANs

# Importamos las librerias
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Fijamos algunos hiperparámteros
batchSize = 64 # Tamaño de bach
imageSize = 64 # Tamaño de las imágenes generadas 64x64

# Creamos las transformaciones para el Generator. Así podremos comparar las imágenes de entrada con las generadas.
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # Descargamos el set de entrenamiento en la caperta ./data y aplicamos la transformación anterior a cada imagen.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # Usamos dataLoader para generar los batches de imágener de set de etrenamiento.

# Definimos la función weights_init que toma como argumento la red neueroanl m, y define  sus pesos iniciales.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# DEfinimos en Generador (G)
class G(nn.Module): # Creamos una clase, G, y dentro definimos el generador. 
# Dentro de la clase definimos la red neuronal y la función 'forward'  que propaga la entrada a lo largo de la red.
    def __init__(self): #Introducimos la función __init__() que definirá la arquitectura del generador.
        super(G, self).__init__() # Lo tomamos de las herramientas de nn.Module. Toma como argumento la clase y el objeto self.
        self.main = nn.Sequential( # Creamos un meta-módulo (main) de una red neuronal que contendrá una sequencia de módulos (convolución, full connections, etc.). Este meta-model que será una propiedad del objeto 'self'
                    nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),  # Capa concolucional inversa que toma un como entrada un vector aleatorio de tamaño 100 y devuelve una imagen como salida. nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # We start with an inversed convolution. Los argumentos que toma son (tamaño del vector de  entrada, número de 'map features' a la salida, tamaño del kernel, stride = 1, padding = 0)
                    nn.BatchNorm2d(512), # Normalizamos todas las features a lo largo del batch.
                    nn.ReLU(True), # Aplicamos la función ReLU para romper la linealidad.
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # Añadimos otra convolución inversa.
                    nn.BatchNorm2d(256), # Normalizamos de nuevo...
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
                    nn.BatchNorm2d(128), 
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                    nn.Tanh(), # Aplicamos la función tanh para romper la linealidad, cuyos valores de y se encuentran entre -1 y +1, que es la estandarización del dataset, y porque además estas imágenes serán la entrada del discriminador y ha de estar estadarizada.
         )
 
                     
    def forward(self, input): # Definimos la función 'forward' que toma como argumentos self(representa la red neuronal en sí) y una entrada (input: el vector aleatorio de tamaño 100) que alimentará la red neuronal, y devolverá una salida  que contiene las imágenes generadas.
        output = self.main(input) # Propagamos la señal a lo largo de toda la red del generador definida por self.main.
        return output # Devolvemos el output con las imágenes generadas.

# Creamos el generador
netG = G() # Creamos el generador (objeto).
netG.apply(weights_init) # Inicializamos los pesos del gnerador (por convenio).


# Definimos el Discriminador(D) que será una red convolucional (imagen -> número). Va a tomar como entrada una imagen generada y devuelve como salida del discriminador un número entre 0 y 1. 
class D(nn.Module): # Creamos una clase para definir el discriminador.
    
    def __init__(self): # Creamos la función __init__() que definirá la arquitectura del discriminador.
        super(D, self).__init__() 
        self.main = nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # Empezamos con la convolución con los argumentos (entrada que son los 3 canales, 64 feature maps, strading = 2 y padding = 1, bias = False) 
            nn.LeakyReLU(0.2, inplace = True), # Esta funcíon se ouede ver aqui https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning. El primer argumento que toma es la pendiente negativa)
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # La salida es 1, número comprendido entre 0 y 1.
            nn.Sigmoid() # Usamos la función de activación sigmoide para romper la linealidad, esta función devuelve valores entre 0 y 1. 0-> Rechaza la imagen , 1->Acepta la imagen. El umbral de aceptación se fija en 0.5
                       )
     
       
    def forward(self, input): # Deinimos la función 'forward' que toma como argumento una entrada que va a limentar a la red y retornará un valor comprendido entre 0 y 1.
        output = self.main(input) # Propagamos la señal a través de la red del discriminador, definida por self.main.
        return output.view(-1) # El -1 se pone para realizar la operación de 'flatten' a la salida  de la convolución para conseguir el resultado.

# Creamos el discriminador.
netD = D() # We create the discriminator object.
netD.apply(weights_init) # We initialize all the weights of its neural network.

# Entrenamos las DCGANs:
    
# El primer paso es actualizar los pesos de la rede neuronal del discriminador. Esto es, entrenar el discriminador para que distinga la realidad de la ficción. 
# a. Lo entrenamos dandole una imagen real y fijamos el objetivo a 1 (imagen real). Hay que maximizar el error calulado por el BCE.
# b. Le pasamos la imagen generada y fijamos el objetivo a 0 (imagen no aceptada).
#    
# El segundo paso consiste en actualizar los pesos del generador. Para ello le alimentamos al discriminador con la imagen que ha creado el generador en el paso anterior. Entonces el discriminador dara un resultado de salida entre 0 y 1 (fijamos el objetivo a 1*). which again will consist of maximizing the area calculated by the bacillus.
# * Aquí siempre se fija a 1 porque este error no se propagará a través del discriminador, sino del generador.
criterion = nn.BCELoss() # Creamos el objeto 'criterion' que mide el error entre el valor predicho y el objetivo. BCE -> Binary Cross Entropy. Asi que el objetivo será 0 cuando queramos reconocer una imagen creada y 1 cuando queramos reconocer una imagen real.
# Necesitamos un optimizador para el generador y otro para el discriminador.
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # Creamos el objeto optimizerD. Argumentos:(parámetros de la red, learning rates y betas).
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(25): #Creamos el bucle para las 25 épocas, a más épocas mejores imágenes. Cada época recorre una vez todo el dataset.

    for i, data in enumerate(dataloader, 0): # Iteramos por las imágenes del dataset minibatch a minibatch y data -> minibatch. enumerate(dataloader, 0) -> para cargar los minibacthes del dataset
        
        #1. El primer paso es actualizar los pesos de la rede neuronal del discriminador. Esto es, entrenar el discriminador para que distinga la realidad de la ficción.
        netD.zero_grad() # Inicializamos a 0 los gradientes del discriminador respecto a los pesos.
        
        #1.1 Lo entrenamos dandole una imagen real y fijamos el objetivo a 1 (imagen real). 
        real, _ = data # Tomamos las imagenes reales del dataset que se van a usar para entrenar el discirminador; data se compone de dos elementos:imagen, etiqueta y nos quedamos con la imagen.
        input = Variable(real) # Lo trasformamos en una variable de torch (tensor, gradiente) para meterlo en la red. real -> imágenes reales.
        target = Variable(torch.ones(input.size()[0])) # Definimos el target del discriminador para las imágenes reales -> 1. Para ello creamos el tensor de dimensiones = minibatch
        output = netD(input) # Definimos la salida y propagamos la señal de la imagen real en la red neuronal del discirminador para tener la respuesta (escalar entre 0 y 1).
        errD_real = criterion(output, target) # Computamos el error entre la predicción (output) y el target ( = 1).
                   
        #1.2 Le pasamos la imagen generada y fijamos el objetivo a 0 (imagen no aceptada).
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # Primero creamos el vector 'noise' o vector de entrada en el generador. Los argumentos que toma son : tamaño del batch, número de elementos, dos dimensiones ficticias para preparar la salida 1,1. Luego hay que introducirlo en una variable de torch.
        fake = netG(noise) # la entrada del discirminador es la salida del generador, en este caso.
        target = Variable(torch.zeros(input.size()[0])) # Definimos el target del discriminador para las imágenes generadas -> 0. Para ello creamos el tensor de dimensiones = minibatch
        output = netD(fake.detach()) # detach es para eliminar el gradiente de la variable de torch y agilizar los cálculos en memoria.
        errD_fake = criterion(output, target) # Computamos el error entre la predicción (output) y el target ( = 0).
        
        # 1.3 Backpropagation del error total
        errD = errD_real + errD_fake # Calculamos el error total
        errD.backward() # Hacemos el bakcpropagation del error a lo largo del discriminador
        optimizerD.step() # Actualizamos los pesos

        # 2. El segundo paso consiste en actualizar los pesos de la red denuroanl del generador. Para ello le alimentamos al discriminador con la imagen que ha creado el generador en el paso anterior. Entonces el discriminador dara un resultado de salida entre 0 y 1 (fijamos el objetivo a 1*). 
        # * Aquí siempre se fija a 1 porque este error no se propagará a través del discriminador, sino del generador y queremos que el generador genere imágenes que el discriminador las interprete como reales.
        netG.zero_grad() # Inicializamos a 0 los gradientes del generador respecto a los pesos.
        target = Variable(torch.ones(input.size()[0])) # El target de todas las imágenes del minibatch hemps dicho que es 1.
        output = netD(fake) # Ahora necesitamos la salida del discriminador cuando la entrada son imágenes generadas. Ahora si tenemos en cuenta el gradiente de 'fake image' porque con esto vamos a actualizar los pesos del generador.
        errG = criterion(output, target) # Este error de predicción se refiere al generador ya que se retropropagará a través de el.
        errG.backward()
        optimizerG.step() # y usamos el optimizador del generador para actualizar los pesos del generador.
        
        # 3. Imprimir los errores y guardar las imágenes reales y generadas de cada minibatch cada 100 steps.
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
