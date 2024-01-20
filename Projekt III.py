#1 Numeryczne rozwiązanie równania, porównanie metod
import numpy as np
import matplotlib.pyplot as plt

#Parametry dla równania Gompertza
K_gompertz = 100000
r_gompertz = 0.4
x_0 = 10
t_gompertz = np.arange(0, 75, 0.1) 

# Parametry dla równania Verhulsta
l_verhulst = r_gompertz
K_verhulst = K_gompertz
h_verhulst = 0.1
N_0_verhulst = x_0
t_verhulst = np.arange(0, 75, h_verhulst)

x_gompertz = np.zeros(len(t_gompertz))
N_verhulst = np.zeros(len(t_verhulst))

# Warunki początkowe
x_gompertz[0] = x_0
N_verhulst[0] = N_0_verhulst

# Rozwiązanie równania Gompertza numerycznie przy użyciu metody Eulera
for i in range(1, len(t_gompertz)):
    x_gompertz[i] = x_gompertz[i - 1] + 0.1 * r_gompertz * x_gompertz[i - 1] * np.log(K_gompertz / x_gompertz[i - 1])

# Rozwiązanie równania Verhulsta
for i in range(1, len(t_verhulst)):
    N_verhulst[i] = N_verhulst[i - 1] + h_verhulst * l_verhulst * N_verhulst[i - 1] * (1 - N_verhulst[i - 1] / K_verhulst)

# Wykres porównawczy
plt.figure(figsize=(10, 6))
plt.plot(t_gompertz, x_gompertz, label='Gompertz Model', color='darkslateblue')
plt.plot(t_verhulst, N_verhulst, label='Verhulst Model', color='forestgreen')
plt.xlabel('Czas')
plt.ylabel('Objętość guza nowotworowego')
plt.title('Porównanie modeli Gompertza i Verhulsta')
plt.legend()
plt.show()



#2.1 Numeryczne rozwiązanie równań w modelu współzawodnictwa
import numpy as np
import matplotlib.pyplot as plt

# Parametry modelu z podpunktu a
e_1, g_1, h_1 = 1.25, 0.5, 0.1
e_2, g_2, h_2 = 0.5, 0.2, 0.2

# Parametry modelu z podpunktu b
e_1_b, g_1_b, h_1_b = 5, 4, 1 
e_2_b, g_2_b, h_2_b = 5, 8, 4

# Warunki początkowe
initial_conditions = [3, 4]

# Numeryczne rozwiązanie równań różniczkowych metodą Eulera dla modelu z podpunktu a
t = np.arange(0, 10, 0.001)
N_1_a = np.zeros_like(t)
N_2_a = np.zeros_like(t)

N_1_a[0], N_2_a[0] = initial_conditions

for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    dN1_dt = (e_1 - g_1 * (h_1 * N_1_a[i-1] + h_2 * N_2_a[i-1])) * N_1_a[i-1]
    dN2_dt = (e_2 - g_2 * (h_1 * N_1_a[i-1] + h_2 * N_2_a[i-1])) * N_2_a[i-1]
    N_1_a[i] = N_1_a[i-1] + dt * dN1_dt
    N_2_a[i] = N_2_a[i-1] + dt * dN2_dt

# Numeryczne rozwiązanie równań różniczkowych metodą Eulera dla modelu z podpunktu b
t_b = np.arange(0, 10, 0.001)
N_1_b = np.zeros_like(t_b)
N_2_b = np.zeros_like(t_b)

N_1_b[0], N_2_b[0] = initial_conditions

for i in range(1, len(t_b)):
    dt = t_b[i] - t_b[i-1]
    dN1_dt = (e_1_b - g_1_b * (h_1_b * N_1_b[i-1] + h_2_b * N_2_b[i-1])) * N_1_b[i-1]
    dN2_dt = (e_2_b - g_2_b * (h_1_b * N_1_b[i-1] + h_2_b * N_2_b[i-1])) * N_2_b[i-1]
    N_1_b[i] = N_1_b[i-1] + dt * dN1_dt
    N_2_b[i] = N_2_b[i-1] + dt * dN2_dt

# Wykresy
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(t, N_1_a, label='N1')
plt.plot(t, N_2_a, label='N2')
plt.title('Model a)')
plt.xlabel('Czas')
plt.ylabel('Liczebność populacji')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_b, N_1_b, label='N1', color = 'violet')
plt.plot(t_b, N_2_b, label='N2', color = 'darkcyan')
plt.title('Model b)')
plt.xlabel('Czas')
plt.ylabel('Liczebność populacji')
plt.legend()
plt.tight_layout()
plt.show()


# Interpretacja
'''Model z podpunktu a przedstawia populacje utrzymujące się na stałym poziomie, co sugeruje równowagę. 
Model b prezentuje bardziej złożoną dynamikę, z większą zmiennością, 
co może wynikać z większej konkurencji między gatunkami.'''



#2.2 Sporządzenie portretu fazowego
import numpy as np
import matplotlib.pyplot as plt

# Parametry modelu
epsilon = [0.8, 0.4]
gamma = [1, 0.5]
h = [0.3, 0.4]

# Czas
t = np.arange(0, 100, 0.01)

# Warunki początkowe dla portretu fazowego
initial_conditions = np.array([[4, 8], [8, 8], [12, 8]])

# Portret fazowy
plt.figure(figsize=(8, 6))

for condition in initial_conditions:
    N = np.zeros((len(t), 2))
    N[0] = condition
    for i in range(1, len(t)):
        dN1_dt = (epsilon[0] - gamma[0] * (h[0] * N[i-1, 0] + h[1] * N[i-1, 1])) * N[i-1, 0]
        dN2_dt = (epsilon[1] - gamma[1] * (h[0] * N[i-1, 0] + h[1] * N[i-1, 1])) * N[i-1, 1]

        N[i, 0] = N[i-1, 0] + 0.01 * dN1_dt
        N[i, 1] = N[i-1, 1] + 0.01 * dN2_dt

    plt.plot(N[:, 0], N[:, 1], label=f'Warunki początkowe: {condition}')

# Dodanie wektorów kierunkowych
e_1, g_1, h_1 = 0.8, 1, 0.3
e_2, g_2, h_2 = 0.4, 0.5, 0.4

# Zakresy
x0 = np.linspace(0, 14, 15)
y0 = np.linspace(0, 10, 15)
X, Y = np.meshgrid(x0, y0)

dX = -e_1 * X + -g_1 * X * Y
dY = -g_2 * X * Y + -e_2 * Y

plt.quiver(X, Y, dX, dY, color='cadetblue', width=0.004, headlength=4)
plt.title('Portret fazowy - Model współzawodnictwa gatunków')
plt.xlabel('Liczebnosc populacji N1')
plt.ylabel('Liczebnosc populacji N2')
plt.legend()
plt.grid(True)
plt.show()


# Interpretacja
'''Na wykresie znajdują się krzywe fazowe dla różnych warunków początkowych oraz wektory kierunkowe, 
które ilustrują zmiany w liczebności populacji w kierunku gradientu funkcji. 
Portret fazowy ukazuje ewolucję systemu w czasie i daje informacje na temat stabilności, 
zbieżności do punktów równowagi oraz ogólnej dynamiki modelu współzawodnictwa gatunków. 
W tym przypadku krzywe fazowe zbiegają się do punktu, 
co może oznaczać, w długotrwałej perspektywie, że system dąży do ustalonego stanu równowagi, 
w którym proporcje między populacjami są stałe.'''
