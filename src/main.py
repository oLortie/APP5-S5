import numpy as np
import matplotlib.pyplot as plt

def rayleigh_theoretical(N):
    dx = 20 / N
    x = np.arange(0, 20, dx)

    # c)
    sigma2 = [0.25, 1, 4, 9, 16]
    rayleigh = [None] * 5
    plt.figure()

    for i in range(len(sigma2)):
        rayleigh[i] = [None] * N

        for j in range(N):
            rayleigh[i][j] = (x[j] / sigma2[i]) * np.exp((-x[j] ** 2) / (2 * sigma2[i]))

        plt.plot(x, rayleigh[i], label=rf"$\sigma^2$={sigma2[i]}")
    plt.legend()
    plt.title('Rayleigh')


def step_1(N, variance):
    print("===== Étape 1 =====")

    # a)
    dx = 20 / N
    x = []

    error_angle = [0]*N
    error_module = [0]*N
    error_module_t = [0]*N

    for i in range(N):
        x.append(i * dx)
        [u1, u2] = np.random.rand(2)
        error_angle[i] = 2*np.pi*u1
        error_module[i] = np.sqrt(-2*variance*np.log(u2))
        error_module_t[i] = np.random.rayleigh(np.sqrt(variance))

    plt.figure()
    plt.plot(error_angle)
    plt.title("Angle de l'erreur")
    plt.xlabel("Numéro de l'échantillon")
    plt.ylabel("Angle (radians)")

    plt.figure()
    plt.plot(error_module)
    plt.title("Module de l'erreur (variance = " + str(variance) + ")")
    plt.xlabel("Numéro de l'échantillon")
    plt.ylabel("Module (m)")

    # b)
    plt.figure()
    plt.hist(error_angle)
    plt.title("Histogramme de l'angle")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Nombre d'échantillons")

    # d), e) et g)
    plt.figure()
    plt.hist(error_module)
    plt.title("Histogramme du module (variance = " + str(variance) + ")")

    plt.figure()
    plt.hist(error_module_t)
    plt.title("Histogramme du module théorique (variance = " + str(variance) + ")")

    return error_module, error_angle


def step_2(error_module, error_angle, variance):
    print("===== Étape 2 =====")
    plt.figure()
    plt.scatter(error_module, error_angle)
    plt.title("Nuage de points Étape 2 (variance = " + str(variance) + ")")
    plt.xlabel("Module (m)")
    plt.ylabel("Angle (radians)")


def step_3(N, d_zero, phi_zero, error_module, error_angle):
    print("===== Étape 3 =====")
    d = [0]*N
    phi = [0]*N
    for i in range(N):
        d[i] = d_zero + error_module[i]*np.cos(error_angle[i])
        phi[i] = phi_zero + error_module[i]*np.sin(error_angle[i])
        phi[i] = np.radians(phi[i])

    plt.figure()
    plt.scatter(d, phi)
    plt.title("Nuage de points Étape 3 ( D0 = " + str(d_zero) + ", phi_zero = " + str(phi_zero) + ")")
    plt.xlabel("D (m)")
    plt.ylabel("Phi (radians)")


def step_4(N, d_zero, phi_zero, error_module, error_angle):
    print("===== Étape 4 =====")
    d_x = [0]*N
    d_y = [0]*N

    for i in range(N):
        d_x[i] = d_zero * np.cos(np.radians(phi_zero)) + error_module[i] * np.cos(error_angle[i])
        d_y[i] = d_zero * np.sin(np.radians(phi_zero)) + error_module[i] * np.sin(error_angle[i])

    plt.figure()
    plt.plot(d_x)
    plt.title("Dx Étape 4 ( D0 = " + str(d_zero) + ", phi_zero = " + str(phi_zero) + ")")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Dx (m)")

    plt.figure()
    plt.plot(d_y)
    plt.title("Dy Étape 4 ( D0 = " + str(d_zero) + ", phi_zero = " + str(phi_zero) + ")")
    plt.xlabel("Numéro d'itération")
    plt.ylabel("Dy (m)")

    plt.figure()
    plt.scatter(d_x, d_y)
    plt.title("Nuage de points Étape 4 ( D0 = " + str(d_zero) + ", phi_zero = " + str(phi_zero) + ")")
    plt.xlabel("Dx")
    plt.ylabel("Dy")

    return d_x, d_y


def step_5(d_x, d_y, d_zero, phi_zero):
    print("===== Étape 5 =====")
    mean_dx = np.mean(d_x)
    std_dev_dx = np.std(d_x)

    fig, ax1 = plt.subplots()
    ax1.set_title("Étape 5: Histogramme de Dx avec D0 = " + str(d_zero) +  " Phi0 = " + str(phi_zero))
    freq_relative = ax1.hist(d_x)
    ax1.set_ybound(0, 3000)
    ax1.set_ylabel("Nombre d'échantillon")
    ax1.set_xlabel("Distance axiale en X")

    freq_relative_x = [None] * (len(freq_relative[1]) - 1)
    for i in range(len(freq_relative[1]) - 1):
        freq_relative_x[i] = (freq_relative[1][i] + freq_relative[1][i + 1]) / 2
        freq_relative[0][i] = freq_relative[0][i] / 100

    ax2 = ax1.twinx()
    ax2.plot(freq_relative_x, freq_relative[0], color='r')
    ax2.set_ybound(0, 30)
    ax2.set_ylabel("Fréquences relatives (%)", color='r')
    ax2.grid()

    print("Moyenne échantillon de Dx: ", mean_dx)
    print("Écart-type échantillon de Dx: ", std_dev_dx)

    mean_dy = np.mean(d_y)
    std_dev_dy = np.std(d_y)

    fig, ax1 = plt.subplots()
    ax1.set_title("Étape 5: Histogramme de Dy avec D0 = " + str(d_zero) +  " Phi0 = " + str(phi_zero))
    freq_relative = ax1.hist(d_y)
    ax1.set_ybound(0, 3000)
    ax1.set_ylabel("Nombre d'échantillon")
    ax1.set_xlabel("Distance axiale en Y")

    freq_relative_x = [None] * (len(freq_relative[1]) - 1)
    for i in range(len(freq_relative[1]) - 1):
        freq_relative_x[i] = (freq_relative[1][i] + freq_relative[1][i + 1]) / 2
        freq_relative[0][i] = freq_relative[0][i] / 100

    ax2 = ax1.twinx()
    ax2.plot(freq_relative_x, freq_relative[0], color='r')
    ax2.set_ybound(0, 30)
    ax2.set_ylabel("Fréquences relatives (%)", color='r')
    ax2.grid()

    print("Moyenne échantillon de Dy: ", mean_dy)
    print("Écart-type échantillon de Dy: ", std_dev_dy)


def step_8(N, d_x, d_y):
    print("===== Étape 8 =====")
    mean_x = np.mean(d_x)
    mean_y = np.mean(d_y)

    d_x_null_mean = d_x - mean_x
    d_y_null_mean = d_y - mean_y

    x_matrix = [d_x_null_mean, d_y_null_mean]

    c_matrix = 1/N*np.matmul(x_matrix, np.transpose(x_matrix))

    print(c_matrix)
    return c_matrix


def step_9(c_matrix):
    print("===== Étape 9 =====")
    S = 5.9915
    T = c_matrix[0][0] + c_matrix[1][1]
    detC = c_matrix[0][0] * c_matrix[1][1] - c_matrix[1][0] * c_matrix[0][1]

    lambda1 = T / 2 + np.sqrt((T ** 2) / 4 - detC)
    lambda2 = T / 2 - np.sqrt((T ** 2) / 4 - detC)

    e1 = [c_matrix[1][0], lambda1 - c_matrix[0][0]]
    e2 = [lambda2 - c_matrix[1][1], c_matrix[0][1]]

    L1 = 2 * np.sqrt(lambda1 * S)
    L2 = 2 * np.sqrt(lambda2 * S)

    teta1 = np.arctan(e1[1] / e1[0])
    teta2 = np.arctan(e2[1] / e2[0])

    print('L1 = ' + str(L1))
    print('L2 = ' + str(L2))
    print('Teta 1 = ' + str(teta1))
    print('Teta 2 = ' + str(teta2))


if __name__ == '__main__':
    N = 10000

    variance1 = 4
    variance2 = 16
    d_zero1 = 50
    d_zero2 = 100
    phi_zero1 = 15
    phi_zero2 = 30

    rayleigh_theoretical(N)

    error_module1, error_angle1 = step_1(N, variance1)
    # error_module2, error_angle2 = step_1(N, variance2)

    step_2(error_module1, error_angle1, variance1)
    #step_2(error_module2, error_angle2, variance2)

    step_3(N, d_zero1, phi_zero1, error_module1, error_angle1)
    # step_3(N, d_zero1, phi_zero2, error_module1, error_angle1)
    # step_3(N, d_zero2, phi_zero1, error_module2, error_angle2)
    # step_3(N, d_zero2, phi_zero2, error_module2, error_angle2)

    d_x1, d_y1 = step_4(N, d_zero1, phi_zero1, error_module1, error_angle1)
    # d_x2, d_y2 = step_4(N, d_zero1, phi_zero2, error_module1, error_angle1)
    # d_x3, d_y3 = step_4(N, d_zero2, phi_zero1, error_module2, error_angle2)
    # d_x4, d_y4 = step_4(N, d_zero2, phi_zero2, error_module2, error_angle2)

    step_5(d_x1, d_y1, d_zero1, phi_zero1)
    # step_5(d_x2, d_y2, d_zero1, phi_zero2)
    # step_5(d_x3, d_y3, d_zero2, phi_zero1)
    # step_5(d_x4, d_y4, d_zero2, phi_zero2)

    c_matrix_1 = step_8(N, d_x1, d_y1)
    # c_matrix_2 = step_8(N, d_x2, d_y2)
    # c_matrix_3 = step_8(N, d_x3, d_y3)
    # c_matrix_4 = step_8(N, d_x4, d_y4)

    step_9(c_matrix_1)
    # step_9(c_matrix_2)
    # step_9(c_matrix_3)
    # step_9(c_matrix_4)

    plt.show()
