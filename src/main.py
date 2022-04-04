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
    plt.title("Angle de l'erreur (variance = " + str(variance) + ")")
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
    plt.title("Histogramme de l'angle (variance = " + str(variance) + ")")
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
        phi[i] = np.radians(phi_zero) + error_module[i]*np.sin(error_angle[i])

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
        d_x[i] = d_zero*np.cos(np.radians(phi_zero)) + error_module[i]*np.cos(error_angle[i])
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


# TODO : Il manque la courbe de fréquence relative à afficher
def step_5(d_x, d_y):
    print("===== Étape 5 =====")
    mean_dx = np.mean(d_x)
    std_dev_dx = np.std(d_x)

    plt.figure()
    plt.hist(d_x)
    plt.title("Histogramme de Dx")

    print("Moyenne échantillon de Dx: ", mean_dx)
    print("Écart-type échantillon de Dx: ", std_dev_dx)

    mean_dy = np.mean(d_y)
    std_dev_dy = np.std(d_y)

    plt.figure()
    plt.hist(d_y)
    plt.title("Histogramme de Dy")

    print("Moyenne échantillon de Dy: ", mean_dy)
    print("Écart-type échantillon de Dy: ", std_dev_dy)


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
    # step_2(error_module2, error_angle2, variance2)

    step_3(N, d_zero1, phi_zero1, error_module1, error_angle1)
    # step_3(N, d_zero1, phi_zero2, error_module1, error_angle1)
    # step_3(N, d_zero2, phi_zero1, error_module2, error_angle2)
    # step_3(N, d_zero2, phi_zero2, error_module2, error_angle2)

    d_x1, d_y1 = step_4(N, d_zero1, phi_zero1, error_module1, error_angle1)
    # d_x2, d_y2 = step_4(N, d_zero1, phi_zero2, error_module1, error_angle1)
    # d_x3, d_y3 = step_4(N, d_zero2, phi_zero1, error_module2, error_angle2)
    # d_x4, d_y4 = step_4(N, d_zero2, phi_zero2, error_module2, error_angle2)

    step_5(d_x1, d_y1)
    # step_5(d_x2, d_y2)
    # step_5(d_x3, d_y3)
    # step_5(d_x4, d_y4)

    plt.show()
