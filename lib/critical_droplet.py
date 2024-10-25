import numpy as np
from lib import renormalization

def critical_droplet_size(r, h_input, qlimit=1000):
    # 1/r is the ratio of T/Tc

    q_results = []
    L_results = []

    Mn = [1, 0, 1, 0]

    q = 3
    while q <= qlimit:

        potts = renormalization.PottsRenormalizationGroup(q)
        j = r * potts.Jc
        h = h_input

        n = -1
        Ln = [0]
        while True:

            n = n + 1
            Ln.append(n)

            U = potts.neigen * np.dot(potts._recursion_matrix_1(j, h), 1)
            Eab, Ebb, Ebc = potts._energy_ab_1(j, h), potts._energy_bb_1(j, h), potts._energy_bc_1(j, h)
            for i in range(n):
                U = potts.neigen * np.dot(potts._recursion_matrix(Eab, Ebb, Ebc), U)
                Eab, Ebb, Ebc = potts._energy_ab(Eab, Ebb, Ebc), potts._energy_bb(Eab, Ebb, Ebc), potts._energy_bc(Eab, Ebb, Ebc)
            M = np.dot(Mn, U)
            magnetization = (M[2] - 1/q) / (1 - 1/q)

            if magnetization < 0:
                pass
            else:
                break

        q_results.append(q)
        L_results.append(potts.b ** ((Ln[-1] + Ln[-2]) / 2))
        q = q + 1

    return np.array(q_results), np.array(L_results)

def maximum_critical_droplet_size(r):
    # 1/c is the ratio of T/Tc

    h_results = []
    L_results = []

    Mn = [1, 0, 1, 0]

    q = 1e6

    k = -8
    h = 10**k
    while h <= 4.5:

        potts = renormalization.PottsRenormalizationGroup(q)
        j = r * potts.Jc

        n = -1
        Ln = [0]
        while True:

            n = n + 1
            Ln.append(n)

            U = potts.neigen * np.dot(potts._recursion_matrix_1(j, h), 1)
            Eab, Ebb, Ebc = potts._energy_ab_1(j, h), potts._energy_bb_1(j, h), potts._energy_bc_1(j, h)
            for i in range(n):
                U = potts.neigen * np.dot(potts._recursion_matrix(Eab, Ebb, Ebc), U)
                Eab, Ebb, Ebc = potts._energy_ab(Eab, Ebb, Ebc), potts._energy_bb(Eab, Ebb, Ebc), potts._energy_bc(Eab, Ebb, Ebc)
            M = np.dot(Mn, U)
            magnetization = (M[2] - 1/q) / (1 - 1/q)

            if magnetization < 0:
                pass
            else:
                break

        h_results.append(h)
        L_results.append(potts.b ** ((Ln[-1] + Ln[-2]) / 2))

        h = h + 10**(k - 2)

        if round(h, 8)   == 0.00000001 and h >= 0.00000001 and h < 0.00000001 + 10**(k-2):
            k = k + 1
        elif round(h, 7) == 0.0000001 and h >= 0.0000001 and h < 0.0000001 + 10**(k-2):
            k = k + 1
        elif round(h, 6) == 0.000001 and h >= 0.000001 and h < 0.000001 + 10**(k-2):
            k = k + 1
        elif round(h, 5) == 0.00001 and h >= 0.00001 and h < 0.00001 + 10**(k-2):
            k = k + 1
        elif round(h, 4) == 0.0001 and h >= 0.0001 and h < 0.0001 + 10**(k-2):
            k = k + 1
        elif round(h, 3) == 0.001 and h >= 0.001 and h < 0.001 + 10**(k-2):
            k = k + 1
        elif round(h, 2) == 0.01 and h >= 0.01 and h < 0.01 + 10**(k-2):
            k = k + 1
        elif round(h, 2) == 0.1 and h >= 0.1 and h < 0.1 + 10**(k-2):
            k = k + 1
        elif round(h, 2) == 1.0 and h >= 1.0 and h < 1.0 + 10**(k-2):
            k = k + 1
        else:
            pass

    return np.array(h_results), np.array(L_results)