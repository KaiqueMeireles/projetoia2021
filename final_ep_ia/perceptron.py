import numpy as np
import pandas as pd
import math as m
import time
import datetime

# le o arquivo de configuracoes
# com o nome do arquivo de treinamento e outros parametros nao aleatorios
def readConfig(file):
    configF = pd.read_csv(file, header=None)
    fileName = configF[0][0]
    tFile = pd.read_csv("training/" + configF[0][0], header=None).T
    inputN = configF[0][1]
    labelN = configF[0][2]
    hidden = configF[0][3]
    hiddenN = configF[0][4]
    epoch = configF[0][5]
    learningRate = configF[0][6]

    return fileName, tFile, int(inputN), int(labelN), hidden, int(hiddenN), int(epoch), float(learningRate)

# cria matrizes de pesos
def createWeightsMatrixes(input, labels):
    return np.random.uniform(-1, 1, size=(labels,input))

# cria a matriz de pesos da camada escondida com n pesos para m neuronios
# n sendo o valor de entradas, no caso dos caracteres 63
# e m o valor de neuronios na camada escondida, definido no arquivo de configuracoes
# e uma matriz para a camada de saida, com m pesos para i neuronios de saida
def createHiddenWeightsMatrixes(input, labels, hiddenN):
    hiddenWeights = np.random.uniform(-1, 1, size=(hiddenN,input))
    outputWeights = np.random.uniform(-1, 1, size=(labels, hiddenN))

    return hiddenWeights, outputWeights

# soma a multiplicacao dos pesos pelos dados de entrada e o bias
def sumWeights(training_data, weights, bias):
    multiplication = np.multiply(training_data, weights)
    y = np.sum(multiplication) + bias

    return y

# retorna 1 se o valor da soma e maior ou igual a 1
# ou retorna 0
def stepActivationFunction(y):
    if y >= 1:
        return 1
    else:
        return 0

# chama a funcao de ativacao para a soma dos pesos
def output(data, weights, bias):
    y = sumWeights(data, weights, bias)
    
    return stepActivationFunction(y)

# chamando o metodo de soma dos dados com os pesos e soma com bias
#  e retorna o valor da funcao sigmoide com esse resultado
def neuronActivation(data, weights, bias):
    y = sumWeights(data, weights, bias)

    return sigmoid(y)
    # return relu(y)

#  retorno da funcao sigmoide
def sigmoid(y):
    return 1/(1 + m.exp(-y))

def relu(y):
    return max(0, y)

def activationRelu(y):
    if y > 0:
        return 1
    else:
        return 0

# para os neuronios de saida a gente compara o resultado final com o limiar 0.5
# para definir se retornamos 1 ou 0
def activationFunction(y):
    if y >= 0.5:
        return 1
    else:
        return 0

#  ajusta os pesos para o perceptron simples
def adjustWeights(learningRate, weights, data, bias, label, output):
    error = label - output
    newWeights = weights + learningRate * data * error
    newBias = bias + learningRate * error
    
    return newWeights, newBias

def perceptron(fileName, tFile, inputN, labelN, hidden, hiddenN, epoch, learningRate):
    # caso a variavel no arquivo de configuracao hidden seja verdadeira, realiza o mlp 
    # caso contrario, roda o perceptron simples
    if hidden == 'true':
        # cria um bias para a camada escondida
        # no caso estamos considerando apenas uma camada escondida
        bias = np.random.uniform(-1, 1, size=(labelN))
        bias_oculto = np.random.uniform(-1, 1, size=(hiddenN))

        # cria ambas as matrizes de pesos aleatorios
        hiddenWeights, outputWeights = createHiddenWeightsMatrixes(inputN, labelN, hiddenN)
        
        hiddenFileName = "hidden-weights" + "-" +  fileName + "-" + str(epoch) + "-" + str(hiddenN) + "-" + str(learningRate) + ".csv"
        np.savetxt(hiddenFileName, hiddenWeights, delimiter = ",")
        with open(hiddenFileName,'a') as fd:
            fd.write("Pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
        with open(hiddenFileName, 'ab') as fd:    
            np.savetxt(fd, bias_oculto, delimiter = ",")
            fd.write("Vies e pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n\n')

        outputFileName = "output-weights" + "-" +  fileName + "-" + str(epoch) + "-" + str(hiddenN) + "-" + str(learningRate) + ".csv"
        np.savetxt(outputFileName, outputWeights, delimiter = ",")
        with open(outputFileName,'a') as fd:
            fd.write("Pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
        with open(outputFileName, 'ab') as fd:  
            np.savetxt(fd, bias, delimiter = ",")
            fd.write("Vies e pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n\n')

        # para cada epoca
        for i in range(epoch):
            # para cada linha do arquivo de teste
            errors = np.empty(labelN)
            # aqui usamos columns pq o pandas importa o .csv como dataframe
            # e cada coluna vira uma linha, por isso, inclusive que no readConfig()
            # chamamos o metodo .T para ler o arquivo de entrada
            for n in range(len(tFile.columns)):
                # cria arrays vazias para guardar os resultados de cada neuronio
                hiddenY = np.empty(hiddenN)
                finalY = np.empty(labelN)
                
                ############ INICIO FEEDFORWARD ############

                # separa a linha em dados de treinamento e resultado esperado
                # ou seja, as 63 primeiras entradas sao colocadas no
                # training_data e os ultimos 7, de resultado, no desired_label
                training_data = np.array(tFile[0:inputN][n])
                desired_label = np.array(tFile[inputN:][n])

                # para cada neuronio na camada escondida
                for hiddenLayer in range(hiddenN):
                    # calcula o valor do neuronio com o training data, com os pesos de camadas
                    # escondidas e guarda na array vazia criada acima
                    hiddenY[hiddenLayer] = neuronActivation(training_data, hiddenWeights[hiddenLayer], bias_oculto[hiddenLayer])
                
                # para cada neuronio na camada de saida
                for outputLayer in range(labelN):
                    # calcula o valor do neuronio com a array de resultados da camada escondida
                    # com os pesos da camada de saida e guarda na array tbm criada acima
                    finalY[outputLayer] = neuronActivation(hiddenY, outputWeights[outputLayer], bias[outputLayer])

                ############ FIM FEEDFORWARD ############
                ############ INICIO BACKPROPAGATION ############

                # para cada neuronio de saida
                for outputLayer in range(labelN): 
                    # calcula o erro do resultado guardado na array com o resultado esperado definido
                    # e guarda ele na array de erros para calculo do erro medio
                    erro = finalY[outputLayer] - desired_label[outputLayer]
                    errors[outputLayer] = erro
                    # calcula a derivada da funcao sigmoide
                    derivada = finalY[outputLayer]*(1.0-finalY[outputLayer])
                    
                    # calcula o delta dos neuronios de saida
                    delta_saida = erro * derivada

                    # para cada neuronio da camada escondida
                    for hiddenLayer in range(hiddenN):
                        # calcula a soma dos pesos do neuronio de saida com o delta de saida
                        weightSum = 0
                        for out in range(len(outputWeights[outputLayer])):
                            weightSum += outputWeights[outputLayer][out] * delta_saida

                        # para entao calcular a derivada da funcao sigmoide do resultado do neuronio escondido
                        derivada = hiddenY[hiddenLayer]*(1.0-hiddenY[hiddenLayer])

                        # para entao calcular o delta do neuronio da camada escondida
                        delta_oculto = weightSum * derivada

                        # e entao altera os bias e o peso desse neuronio da camada escondida com
                        # o delta_oculto calculado acima
                        bias_oculto[hiddenLayer] = bias_oculto[hiddenLayer] + (learningRate * delta_oculto)
                        for hid in range(len(hiddenWeights[hiddenLayer])):
                            hiddenWeights[hiddenLayer][hid] = hiddenWeights[hiddenLayer][hid] + (learningRate * delta_oculto * hiddenY[hiddenLayer])
                
                    # e entao altera os bias e o peso desse neuronio da camada de saida com
                    # o delta de saida calculado acima
                    bias[outputLayer] = bias[outputLayer] + (learningRate * delta_saida)
                    for out in range(len(outputWeights[outputLayer])):
                        outputWeights[outputLayer][out] = outputWeights[outputLayer][out] + (learningRate * delta_saida * finalY[outputLayer])
                
                ############ FIM BACKPROPAGATION ############
            
            # criacao do arquivo para guardar os erros das epocas
            sqFileName = "square-avg-error" + "-" +  fileName + "-" + str(epoch) + "-" + str(hiddenN) + "-" + str(learningRate) + ".csv"
            with open(sqFileName, "ab") as fd:
                fd.write(str(i) + "," + str(np.sum(errors)/len(errors)) + "\n")

        with open(hiddenFileName, "ab") as fd:
            np.savetxt(fd, hiddenWeights, delimiter = ',')
            fd.write("Pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
            np.savetxt(fd, bias_oculto, delimiter = ",")
            fd.write("Vies e pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))

        with open(outputFileName, "ab") as fd:
            np.savetxt(fd, outputWeights, delimiter = ',')
            fd.write("Pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
            np.savetxt(fd, bias, delimiter = ",")
            fd.write("Vies e pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))

        # escolhe o arquivo de teste
        if fileName == "caracteres-limpo.csv":
            testFile = pd.read_csv("test/caracteres-ruido.csv", header=None).T
        else:
            testFile = pd.read_csv("test/problemXOR.csv", header=None).T
        
        resultsHidden = np.empty(hiddenN)
        resultsFinal = np.empty(labelN)

        # para cada linha
        for i in range(len(testFile.columns)):
            test_data = np.array(testFile[i][0:inputN])
            
            for i in range(hiddenN):
                # chama a funcao do neuronio para a camada escondida com os pesos finais criados
                # acima, no treinamento
                # e guarda o valor para cada neuronio
                resultsHidden[i] = neuronActivation(test_data, hiddenWeights[i], bias_oculto[i])

            for j in range(labelN):
                # chama a funcao do neuronio para a camada de saida com os pesos finais criados
                # acima, no treinamento
                # e guarda o valor para cada neuronio
                y = neuronActivation(resultsHidden, outputWeights[j], bias[j])
                resultsFinal[j] = activationFunction(y)
                
            
            resultsFileName = "results" + "-" +  fileName + "-" + str(epoch) + "-" + str(hiddenN) + "-" + str(learningRate) + ".csv"

            with open(resultsFileName, "ab") as fd:
                np.savetxt(fd, resultsFinal, delimiter = ',')
                fd.write("Resultados obtidos em: \n")
                fd.write(str(datetime.datetime.now()))
                fd.write("\n")
    else:# se nao tiver camada escondidas, executa o perceptron simples

        # cria a matriz de pesos do perceptron simples
        weights = createWeightsMatrixes(inputN, labelN)
        bias = np.random.random()

        outputFileName = "weights" + "-" +  fileName + "-" + str(epoch) + "-" + str(hiddenN) + "-" + str(learningRate) + ".csv"
        np.savetxt(outputFileName, weights, delimiter = ",")
        with open(outputFileName,'a') as fd:
            fd.write("Pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
        with open(outputFileName, 'ab') as fd:  
            fd.write(str(bias))
            fd.write("Vies e pesos iniciais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n\n')

        print("Pesos iniciais ", weights)

        # para cada epoca
        for i in range(epoch):
            
            # para cada linha
            for n in range(len(tFile.columns)):
                # separa a linha entre dados de treinamento e valor esperado
                training_data = np.array(tFile[n][0:inputN])
                desired_label = np.array(tFile[n][inputN:])
                
                # calcula o resultado desse dado de treinamento com os pesos
                y = output(training_data, weights, bias)
                
                # ajusta os pesos e bias 
                weights, bias= adjustWeights(learningRate, weights, training_data, bias, desired_label, y)

        with open(outputFileName, "ab") as fd:
            np.savetxt(fd, weights, delimiter = ',')
            fd.write("Pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
            fd.write('\n')
            fd.write(str(bias))
            fd.write("Vies e pesos finais gerados em: \n")
            fd.write(str(datetime.datetime.now()))
        
        print("Pesos finais ", weights)
        
        # escolhe o arquivo de teste
        testFile = pd.read_csv("test/problemAND.csv", header=None).T
        
        # para cada linha
        for i in range(len(testFile.columns)):
            test_data = np.array(testFile[i][0:inputN])
            # calcula os resultados finais com os pesos finais
            resultsFinal = output(test_data, weights, bias)
            print resultsFinal

# INICIALIZA OS PARAMETROS, LENDO DO ARQUIVO DE CONFIGURACAO PARA TREINAMENTO E CRIA UM BIAS ALEATORIO
fileName, tFile, inputN, labelN, hidden, hiddenN, epoch, learningRate = readConfig("configCHA.csv")
perceptron(fileName, tFile, inputN, labelN, hidden, hiddenN, epoch, learningRate)

fileName, tFile, inputN, labelN, hidden, hiddenN, epoch, learningRate = readConfig("configAND.csv")
perceptron(fileName, tFile, inputN, labelN, hidden, hiddenN, epoch, learningRate)