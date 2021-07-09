import numpy as np
import matplotlib.pyplot as p

graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-1000-10-0.3.csv", delimiter=",", names=["epocas", "erro"])
p.plot(graph["epocas"], graph["erro"])
p.savefig("sq-1000-10-0.3.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-1000-20-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-1000-20-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-2000-10-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-2000-10-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-2000-20-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-2000-20-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-2000-20-0.8.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-2000-20-0.8.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-5000-10-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-5000-10-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-5000-20-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-5000-20-0.2.png")

#########
# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-10000-10-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-10000-10-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-10000-10-0.5.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-10000-10-0.5.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-10000-20-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-10000-20-0.2.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-10000-20-0.5.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-10000-20-0.5.png")

# graph = np.genfromtxt("square-avg-error-caracteres-limpo.csv-10000-20-0.8.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sq-10000-20-0.8.png")

# graph = np.genfromtxt("square-avg-error-problemXOR.csv-1000-2-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sqXOR-1000-2-0.2.png")

# graph = np.genfromtxt("square-avg-error-problemXOR.csv-2000-2-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sqXOR-20000-2-0.2.png")

# graph = np.genfromtxt("square-avg-error-problemXOR.csv-5000-2-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sqXOR-5000-2-0.2.png")

# graph = np.genfromtxt("square-avg-error-problemXOR.csv-10000-2-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sqXOR-10000-2-0.2.png")

# graph = np.genfromtxt("square-avg-error-problemAND.csv-1000-0-0.2.csv", delimiter=",", names=["epocas", "erro"])
# p.plot(graph["epocas"], graph["erro"])
# p.savefig("sqAND-10000-10-0.2.png")