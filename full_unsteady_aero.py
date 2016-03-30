import numpy as np
import matplotlib.pylab as plt

cplx = complex


def char_poly(a, xa, ra, mu, omega, u):
    A = 0.0225*mu**2*omega**2*ra**2*u**2 + (-0.0225 - 0.045*a)*mu*omega**2*u**4
    B = 0.3*mu**2*omega**2*ra**2*u + (-0.2325 + (-0.4875 + 0.045*a)*a)*mu*omega**2*u**3 + 0.045*mu*ra**2*u**3
    C = (-0.5134375000000001 + (-1.4 + 0.4875*a)*a)*mu*omega**2*u**2 + 0.0225*mu**2*ra**2*u**2 + 0.0225*u**4 + mu**2*omega**2*ra**2*(1. + 0.0225*u**2) + mu*(0.4875*ra**2*u**2 + u**4*(-0.0225 - 0.045*a - 0.045*xa))
    D = (0.26249999999999996 + a*(-1. + 1.4*a))*mu*omega**2*u + 0.3*mu**2*ra**2*u + 0.3*mu**2*omega**2*ra**2*u + (0.23812500000000014 + 2.842170943040401e-16*a)*u**3 + mu*(ra**2*u*(1.4 + 0.045*u**2) + u**3*(-0.2325 + a*(-0.4875 + 0.045*a + 0.09*xa) - 0.4875*xa))
    E = (0.125 + 1.*a**2)*mu*omega**2 + 1.*mu**2*omega**2*ra**2 + 0.6446875*u**2 + mu**2*ra**2*(1. + 0.0225*u**2) + mu*(ra**2*(1. + 0.4875*u**2) + u**2*(-0.5134375000000001 + a*(-1.4 + 0.4875*a + 0.975*xa) - 1.4*xa)) - 0.0225*mu**2*u**2*xa**2
    F = ((0.380435 + 1.11022*10**-16*a)*u + mu*u*(0.26087 + 0.956522*ra**2 - xa + a*(-1. + 0.956522*a + 1.91304*xa)))
    G = 0.4*u + 0.3*mu**2*ra**2*u - 0.3*mu**2*u*xa**2 + mu*u*(0.26249999999999996 + 1.4*ra**2 - 1.*xa + a*(-1. + 1.4*a + 2.8*xa))
    return G, F, E, D, C, B, A


def compute_flutter(omega, plot=True): 
    a = -.3
    xa = .2 
    ra = .3 
    mu = 10 
    e = .2 

    U = np.linspace(.1,2,200)

    data0 = []
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []

    flutter_speed = -10
    for u in U: 
        r = np.roots(char_poly(a, xa, ra, mu, omega, u))
        data0.append(r[0])
        data1.append(r[1])
        data2.append(r[2])
        data3.append(r[3])
        data4.append(r[4])
        data5.append(r[5])

        if np.max(r.real > 0) and flutter_speed < 0: 
            mode_idx = np.argmax(r.real)
            flutter_speed = u
            flutter_freq = r.imag[mode_idx]

    data0 = np.array(data0, dtype=complex)
    data1 = np.array(data1, dtype=complex)
    data2 = np.array(data2, dtype=complex)
    data3 = np.array(data3, dtype=complex)
    data4 = np.array(data4, dtype=complex)
    data5 = np.array(data5, dtype=complex)

    print "flutter freq: ", flutter_freq
    print "flutter speed: ", flutter_speed
    if plot: 
        colors = plt.cm.viridis(np.linspace(0,1,6))
        fig, ax = plt.subplots()
        ms=15
        ax.scatter(U.real, data0.real, c=colors[0], s=ms, lw=0)
        ax.scatter(U.real, data1.real, c=colors[1], s=ms, lw=0)
        ax.scatter(U.real, data2.real, c=colors[2], s=ms, lw=0)
        ax.scatter(U.real, data3.real, c=colors[3], s=ms, lw=0)
        ax.scatter(U.real, data4.real, c=colors[4], s=ms, lw=0)
        ax.scatter(U.real, data5.real, c=colors[5], s=ms, lw=0)
        ax.axhline(0, ls="-", c="k")
        ax.axvline(flutter_speed, ls="--", c='r')
        ax.set_ylabel(r'$\xi$', fontsize=15, rotation="horizontal", ha='right')
        ax.set_xlabel(r'$\bar{U}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)

        fig, ax = plt.subplots()
        ms = 10
        ax.scatter(U.real, data0.imag, c=colors[0], s=ms, lw=0)
        ax.scatter(U.real, data1.imag, c=colors[1], s=ms, lw=0)
        ax.scatter(U.real, data2.imag, c=colors[2], s=ms, lw=0)
        ax.scatter(U.real, data3.imag, c=colors[3], s=ms, lw=0)
        ax.scatter(U.real, data4.imag, c=colors[4], s=ms, lw=0)
        ax.scatter(U.real, data5.imag, c=colors[5], s=ms, lw=0)
        ax.axvline(flutter_speed, ls="--", c='r')
        ax.set_ylabel(r'$\omega$', fontsize=15, rotation="horizontal", ha='right')
        ax.set_xlabel(r'$\bar{U}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)

        plt.show()

    return flutter_freq, flutter_speed

compute_flutter(.2, True)
print 
compute_flutter(.5, True)
print 
compute_flutter(.8, True)
