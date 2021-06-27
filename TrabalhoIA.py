import numpy as np # Biblioteca utilizada para operações aritméticas/matemáticas no decorrer do código.

def funcAtivacao(net):
    if (net>=0): return 1
    return 0

''' A gente vai ter que mudar esse dataset e pegar direto do excel'''
# Dataset do AND lógico (tabela verdade)
dataset = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# Testes que eu estava fazendo com a matriz
# _________________________________________
#column = dataset[:, 0:dataset.shape[1]-1]
#column = dataset[:, -1]
#print(column[1,])


# man nao sei oq eh esse ETA ai nao<><><><><><><><><><><><><><><><><>IMPORTANTE!!!!!!!!!!<><><><><><><><><><><><><><><><><><><><>
def treinoIA(dataset, eta = 0.1, bias = 1e-3):

    # Chama a biblioteca NumPy para determinar os pesos de maneira aleatória, entre -0.5 e 0.5
    weights = np.random.uniform(-0.5, 0.5, size=(3,1))

    # Pega a última coluna
    classId = dataset[:, -1]

    # Pega as duas primeiras colunas como valores de exemplo
    X = dataset[:, 0:dataset.shape[1]-1]

    # Pega a última coluna como valores esperados
    Y = dataset[:,-1]

# _______ATÉ AQUI TÁ TUDO OK E ENTENDIDO_______

    sqerror = 2*bias

    while(sqerror > bias):
        sqerror = 0

        # para cada linha no conjunto X, pegue um exemplo desse contido em X na linha em i e faça o treinamento para esse exemplo
        for i, valor in enumerate(X):
            print("posicao " + i + "do vetor tem o valor" + valor) # Isso foi eu >tentando< testar como o FOR tava iterando em X, pode deletar se for o caso
            
            # Pega uma linha da matriz feita (essa matriz é a original só que sem a última coluna)
            x = X[i,]
            y = Y[i]

            # Função net (?) para multiplicar as entradas com os pesos e adicionar o theta
            net = [x,1]@weights #ou então net = np.dot([x,1], weights), não sei se o funcionamento vai ser o mesmo
            yObtido = funcAtivacao(net)

            error = y - yObtido
            sqerror = sqerror + error**2

            # Aquela parte que ele começa a falar de derivada:
            #derivadaW1 = 2*(error)*-x[1]
            #derivadaW2 = 2*(error)*-x[2]
            #derivadaTheta = 2*(error)*-1
            derivada = 2*error*(-[x,1])

            weights = weights - eta * derivada

        sqerror = sqerror / X.shape[0]
        print("sqerror = " + sqerror + "\n")

    return weights

    #Não sei como faz pra executar o código todo, mas em teoria tá tudo certo, só vão aparecer os erros quando executarmos