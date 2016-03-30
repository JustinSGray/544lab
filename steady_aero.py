import numpy as np
import matplotlib.pylab as plt

cplx = complex


def r1(a, xa, ra, mu, e, omega, u):
    val = (-0.7071067811865476*np.sqrt((-1.*(1.*mu*ra**2 + 1.*mu*omega**2*ra**2 - 2.*e*u**2 - 2.*u**2*xa + 1.*np.sqrt((mu*(1 + omega**2)*ra**2 + u**2*(-2.*e - 2.*xa))**2 - 4.*mu*omega**2*(mu*ra**2 - 2.*e*u**2)*(ra**2 - 1.*xa**2))))/(mu*(ra**2 - 1.*xa**2))))
    return val 


def r2(a, xa, ra, mu, e, omega, u):
    val = (0.7071067811865476*np.sqrt((-1.*(1.*mu*ra**2 + 1.*mu*omega**2*ra**2 - 2.*e*u**2 - 2.*u**2*xa + 1.*np.sqrt((mu*(1 + omega**2)*ra**2 + u**2*(-2.*e - 2.*xa))**2 - 4.*mu*omega**2*(mu*ra**2 - 2.*e*u**2)*(ra**2 - 1.*xa**2))))/(mu*(ra**2 - 1.*xa**2))))
    return val 


def r3(a, xa, ra, mu, e, omega, u):
    val = (-0.7071067811865476*np.sqrt((-1.*(1.*mu*ra**2 + 1.*mu*omega**2*ra**2 - 2.*e*u**2 - 2.*u**2*xa - 1.*np.sqrt((mu*(1 + omega**2)*ra**2 + u**2*(-2.*e - 2.*xa))**2 - 4.*mu*omega**2*(mu*ra**2 - 2.*e*u**2)*(ra**2 - 1.*xa**2))))/(mu*(ra**2 - 1.*xa**2))))
    return val 


def r4(a, xa, ra, mu, e, omega, u):
    val = (0.7071067811865476*np.sqrt((-1.*(1.*mu*ra**2 + 1.*mu*omega**2*ra**2 - 2.*e*u**2 - 2.*u**2*xa - 1.*np.sqrt((mu*(1 + omega**2)*ra**2 + u**2*(-2.*e - 2.*xa))**2 - 4.*mu*omega**2*(mu*ra**2 - 2.*e*u**2)*(ra**2 - 1.*xa**2))))/(mu*(ra**2 - 1.*xa**2))))
    return val 


def compute_flutter(omega, plot=True): 
    a = -.3
    xa = .2 
    ra = .3 
    mu = 10 
    e = .2 

    a = -.3
    xa = .2 
    ra = .3 
    mu = 10 
    e = .2 

    U = np.linspace(1e-4,2.0,1000, dtype="complex")
    data1 = r1(a, xa, ra, mu, e, omega, U)
    data2 = r2(a, xa, ra, mu, e, omega, U)
    data3 = r3(a, xa, ra, mu, e, omega, U)
    data4 = r4(a, xa, ra, mu, e, omega, U)

    f1 = data1.real[4:] > 1e-4
    if np.any(f1): 
        f1 = np.argmax(f1)
    else: 
        f1 = len(U)-1

    f2 = data2.real[4:] > 1e-4
    if np.any(f2): 
        f2 = np.argmax(f2)
    else: 
        f2 = len(U)-1

    f3 = data3.real[4:] > 1e-4
    if np.any(f3): 
        f3 = np.argmax(f3)
    else: 
        f3 = len(U)-1

    f4 = data4.real[4:] > 1e-4
    if np.any(f4): 
        f4 = np.argmax(f4)
    else: 
        f4 = len(U)-1
    
    idxs = np.array([f1, f2, f3, f4]) 
    f_idx = np.min(idxs)
    mode_idx = np.argmin(idxs)
    
    flutter_speed = U.real[f_idx]
    if mode_idx == 0: 
        flutter_freq = data1.imag[f_idx]
    if mode_idx == 1: 
        flutter_freq = data2.imag[f_idx]
    if mode_idx == 2: 
        flutter_freq = data3.imag[f_idx]
    if mode_idx == 3: 
        flutter_freq = data4.imag[f_idx]  

    flutter_freq = np.abs(flutter_freq)
    print "flutter freq: ", flutter_freq
    print "flutter speed: ", flutter_speed

    if plot: 
        colors = plt.cm.viridis(np.linspace(0,1,4))
        ms = 10

        fig, ax = plt.subplots()
        ax.scatter(U.real, data1.real, c=colors[0], s=ms, lw=0)
        ax.scatter(U.real, data2.real, c=colors[1], s=ms, lw=0)
        ax.scatter(U.real, data3.real, c=colors[2], s=ms, lw=0)
        ax.scatter(U.real, data4.real, c=colors[3], s=ms, lw=0)
        ax.axhline(0, ls="-", c="k")
        ax.axvline(flutter_speed, ls="--", c='r')
        ax.set_ylabel(r'$\xi$', fontsize=15, rotation="horizontal", ha='right')
        ax.set_xlabel(r'$\bar{U}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)
        ax.set_ylim(-1,1)

        fig, ax = plt.subplots()
        ax.scatter(U.real, data1.imag, c=colors[0], s=ms, lw=0)
        ax.scatter(U.real, data2.imag, c=colors[1], s=ms, lw=0)
        ax.scatter(U.real, data3.imag, c=colors[2], s=ms, lw=0)
        ax.scatter(U.real, data4.imag, c=colors[3], s=ms, lw=0)
        ax.axvline(flutter_speed, ls="--", c='r')

        ax.set_ylabel(r'$\omega$', fontsize=15, rotation="horizontal", ha='right')
        ax.set_xlabel(r'$\bar{U}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)
        plt.show()

    return flutter_freq, flutter_speed


compute_flutter(.2, False)
print 
compute_flutter(.5, False)
print 
compute_flutter(.8, False)
