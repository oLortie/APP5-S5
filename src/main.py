import numpy as np
import matplotlib.pyplot as plt


def etape_1(N, variance):
    print("===== Étape 1 =====")

    # a)
    error_angle = [0]*N
    error_module = [0]*N
    error_module_t = [0]*N
    for i in range(N):
        [u1, u2] = np.random.rand(2)
        error_angle[i] = 2*np.pi*u1
        error_module[i] = np.sqrt(-2*variance*np.log(u2))
        error_module_t[i] = np.random.rayleigh(np.sqrt(variance))

    # b)
    plt.figure()
    plt.hist(error_angle)
    plt.title("Histogramme de l'angle")

    # c)
    # x_rayleigh = np.arange(0, 20, 20/N)
    # rayleigh = [0]*N
    #
    # for i in range(len(x_rayleigh)):
    #     rayleigh[i] = (x_rayleigh/variance)*np.exp(-x_rayleigh**2/(2*variance))
    #
    # plt.figure()
    # plt.plot(x_rayleigh, rayleigh)
    # plt.title("Courbe théorique de Rayleigh")

    # d), e) et g)
    plt.figure()
    plt.hist(error_module)
    plt.title("Histogramme du module")

    plt.figure()
    plt.hist(error_module_t)
    plt.title("Histogramme du module théorique")

    # f) : Aucune idée quoi faire

    return error_module, error_angle

def etape_2(error_module, error_angle):
    print("===== Étape 2 =====")
    plt.figure()
    plt.scatter(error_module, error_angle)
    plt.title("Nuage de points (Étape 2)")
    plt.xlabel("Module")
    plt.ylabel("Angle")

def etape_3(N, d_zero, phi_zero, error_module, error_angle):
    print("===== Étape 3 =====")
    d = [0]*N
    phi = [0]*N
    for i in range(N):
        d[i] = d_zero + error_module[i]*np.cos(error_angle[i])
        phi[i] = phi_zero + error_module[i]*np.sin(error_angle[i])

    plt.figure()
    plt.scatter(d, phi)
    plt.title("Nuage de points (Étape 3)")
    plt.xlabel("D")
    plt.ylabel("Phi")

def etape_4(N, d_zero, phi_zero, error_module, error_angle):
    print("===== Étape 4 =====")
    d_x = [0]*N
    d_y = [0]*N

    for i in range(N):
        d_x[i] = d_zero*np.cos(np.radians(phi_zero)) + error_module[i]*np.cos(error_angle[i])
        d_y[i] = d_zero * np.sin(np.radians(phi_zero)) + error_module[i] * np.sin(error_angle[i])

    plt.figure()
    plt.scatter(d_x, d_y)
    plt.title("Nuage de points (Étape 4)")
    plt.xlabel("Dx")
    plt.ylabel("Dy")

    return d_x, d_y


def etape_5(d_x, d_y):
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
    variance = 16 # Peut être porté à changer
    d_zero = 100 # Peut être porté à changer
    phi_zero = 15

    error_module, error_angle = etape_1(N, variance)
    etape_2(error_module, error_angle)
    etape_3(N, d_zero, phi_zero, error_module, error_angle)
    d_x, d_y = etape_4(N, d_zero, phi_zero, error_module, error_angle)
    etape_5(d_x, d_y)

    plt.show()
