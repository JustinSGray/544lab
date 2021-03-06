import numpy as np
import matplotlib.pylab as plt

cplx = complex

def r1(a, xa, ra, mu, omega, k): 
    # print (4.012350061704833e35*mu*omega**2 + 8.024700123409667e35*a*mu*omega**2 + 
    #       cplx(9.807966817500704e36,-1.20370501851145e36)*k*mu*omega**2 + cplx(1.9615933635001409e37,-2.0061750308524168e36)*a*k*mu*omega**2 - 
    #       cplx(0.,8.024700123409667e35)*a**2*k*mu*omega**2 + (1.8484673825937405e37,4.903983408750352e36)*k**2*mu*omega**2 + 
    #       3.5665333881820743e37*a*k**2*mu*omega**2 - cplx(2.0061750308524168e36,1.9615933635001409e37)*a**2*k**2*mu*omega**2 + 
    #       cplx(4.359096363333646e38,-6.2414334293186295e37)*k**3*mu*omega**2 + cplx(8.718192726667292e38,-7.133066776364149e37)*a*k**3*mu*omega**2 - 
    #       cplx(0.,3.5665333881820743e37)*a**2*k**3*mu*omega**2 + cplx(3.1207167146593148e37,2.179548181666823e38)*k**4*mu*omega**2 - 
    #       cplx(7.133066776364149e37,8.718192726667292e38)*a**2*k**4*mu*omega**2 - cplx(0.,3.96281487575786e38)*k**5*mu*omega**2 + 
    #       cplx(0.,7.92562975151572e38)*a*k**5*mu*omega**2 + 9.90703718939465e37*k**6*mu*omega**2 + 7.92562975151572e38*a**2*k**6*mu*omega**2)

    # exit()
    val = (4.012350061704833e35*mu*omega**2 + 8.024700123409667e35*a*mu*omega**2 + 
          cplx(9.807966817500704e36,-1.20370501851145e36)*k*mu*omega**2 + cplx(1.9615933635001409e37,-2.0061750308524168e36)*a*k*mu*omega**2 - 
          cplx(0.,8.024700123409667e35)*a**2*k*mu*omega**2 + cplx(1.8484673825937405e37,4.903983408750352e36)*k**2*mu*omega**2 + 
          3.5665333881820743e37*a*k**2*mu*omega**2 - cplx(2.0061750308524168e36,1.9615933635001409e37)*a**2*k**2*mu*omega**2 + 
          cplx(4.359096363333646e38,-6.2414334293186295e37)*k**3*mu*omega**2 + cplx(8.718192726667292e38,-7.133066776364149e37)*a*k**3*mu*omega**2 - 
          cplx(0.,3.5665333881820743e37)*a**2*k**3*mu*omega**2 + cplx(3.1207167146593148e37,2.179548181666823e38)*k**4*mu*omega**2 - 
          cplx(7.133066776364149e37,8.718192726667292e38)*a**2*k**4*mu*omega**2 - cplx(0.,3.96281487575786e38)*k**5*mu*omega**2 + 
          cplx(0.,7.92562975151572e38)*a*k**5*mu*omega**2 + 9.90703718939465e37*k**6*mu*omega**2 + 7.92562975151572e38*a**2*k**6*mu*omega**2 - 
          cplx(0.,8.024700123409667e35)*k*mu*ra**2 - cplx(2.006175030852417e36,1.9615933635001409e37)*k**2*mu*ra**2 - 
          cplx(0.,3.5665333881820743e37)*k**3*mu*ra**2 - cplx(7.133066776364149e37,8.718192726667292e38)*k**4*mu*ra**2 + 
          7.92562975151572e38*k**6*mu*ra**2 + 4.012350061704833e35*k**2*mu**2*ra**2 + 3.5665333881820743e37*k**4*mu**2*ra**2 + 
          7.92562975151572e38*k**6*mu**2*ra**2 + 4.012350061704833e35*k**2*mu**2*omega**2*ra**2 + 
          3.5665333881820743e37*k**4*mu**2*omega**2*ra**2 + 7.92562975151572e38*k**6*mu**2*omega**2*ra**2 - 
          0.5*k**2*np.sqrt((mu**2*((k*(cplx(0.,1.6049400246819333e36) + cplx(0.,7.133066776364149e37)*k**2 + 
                       k**5*(-1.585125950303144e39 - 1.585125950303144e39*mu) + 
                       k**3*(cplx(1.4266133552728297e38,1.7436385453334584e39) - 7.133066776364149e37*mu) + 
                       k*(cplx(4.012350061704834e36,3.9231867270002817e37) - 8.024700123409667e35*mu))*ra**2 + 
                    omega**2*(-8.024700123409667e35 - cplx(1.9615933635001409e37,-2.4074100370229e36)*k - 
                       cplx(8.718192726667292e38,-1.2482866858637259e38)*k**3 + cplx(0.,7.92562975151572e38)*k**5 + 
                       a**2*k*(cplx(0.,1.6049400246819333e36) + cplx(4.0123500617048336e36,3.9231867270002817e37)*k + cplx(0.,7.133066776364149e37)*k**2 + 
                          cplx(1.4266133552728297e38,1.7436385453334584e39)*k**3 - 1.585125950303144e39*k**5) + 
                       a*(-1.6049400246819333e36 - cplx(3.9231867270002817e37,-4.0123500617048336e36)*k - 7.133066776364149e37*k**2 - 
                          cplx(1.7436385453334584e39,-1.4266133552728297e38)*k**3 - cplx(0.,1.585125950303144e39)*k**5) + 
                       k**6*(-1.98140743787893e38 - 1.585125950303144e39*mu*ra**2) + 
                       k**4*(cplx(-6.2414334293186295e37,-4.359096363333646e38) - 7.133066776364149e37*mu*ra**2) + 
                       k**2*(cplx(-3.696934765187481e37,-9.807966817500704e36) - 8.024700123409667e35*mu*ra**2)))**2 - 
                 4.*k**2*(8.024700123409667e35 + 7.133066776364149e37*k**2 + 1.585125950303144e39*k**4)*omega**2*ra**2*
                  (-8.024700123409667e35 + 8.024700123409667e35*mu + 
                    a**2*k*(cplx(0.,-1.6049400246819333e36) - cplx(4.0123500617048336e36,3.9231867270002817e37)*k - cplx(0.,7.133066776364149e37)*k**2 - 
                       cplx(1.4266133552728297e38,1.7436385453334584e39)*k**3 + 1.585125950303144e39*k**5)*mu + 
                    k**5*(cplx(0.,-7.92562975151572e38) + mu*(cplx(0.,-7.92562975151572e38) + cplx(0.,1.585125950303144e39)*xa)) + 
                    1.6049400246819333e36*mu*xa + k*(cplx(-1.9615933635001409e37,2.2067925339376582e36) + 
                       mu*(cplx(1.9615933635001409e37,-2.4074100370229e36) - cplx(0.,1.6049400246819333e36)*ra**2 + 
                          cplx(3.9231867270002817e37,-4.0123500617048336e36)*xa)) + 
                    k**3*(cplx(-8.718192726667292e38,8.024700123409667e37) + 
                       mu*(cplx(8.718192726667292e38,-1.2482866858637259e38) - cplx(0.,7.133066776364149e37)*ra**2 + 
                          cplx(1.7436385453334584e39,-1.4266133552728297e38)*xa)) + 
                    a*(2.252599458505449e21 + mu*(1.6049400246819333e36 + cplx(0.,1.585125950303144e39)*k**5 + 
                          k**2*(7.133066776364149e37 - cplx(8.024700123409668e36,7.846373454000563e37)*xa) + 
                          k*(cplx(3.9231867270002817e37,-4.0123500617048336e36) - cplx(0.,3.2098800493638666e36)*xa) + 
                          k**3*(cplx(1.7436385453334584e39,-1.4266133552728297e38) - cplx(0.,1.4266133552728297e38)*xa) - 
                          cplx(2.8532267105456594e38,3.487277090666917e39)*k**4*xa + 3.170251900606288e39*k**6*xa)) + 
                    k**6*(1.98140743787893e38 + mu*(1.98140743787893e38 + 1.585125950303144e39*ra**2) + 
                       mu**2*(1.585125950303144e39*ra**2 - 1.585125950303144e39*xa**2)) + 
                    k**4*(cplx(3.5665333881820743e37,2.179548181666823e38) + 
                       mu*(cplx(6.2414334293186295e37,4.359096363333646e38) - cplx(1.4266133552728297e38,1.7436385453334584e39)*ra**2) + 
                       mu**2*(7.133066776364149e37*ra**2 - 7.133066776364149e37*xa**2)) + 
                    k**2*(cplx(-3.4963172621022395e37,4.903983408750352e36) + 
                       mu*(cplx(3.696934765187481e37,9.807966817500704e36) - cplx(4.012350061704834e36,3.9231867270002817e37)*ra**2 + 
                          7.133066776364149e37*xa) + mu**2*(8.024700123409667e35*ra**2 - 8.024700123409667e35*xa**2)))))/k**4))/(k**2*(8.024700123409667e35 + 7.133066776364149e37*k**2 + 1.585125950303144e39*k**4)*mu**2*omega**2*ra**2)

    return val


def r2(a, xa, ra, mu, omega, k): 
    val = (4.012350061704833e35*mu*omega**2 + 8.024700123409667e35*a*mu*omega**2 + cplx(9.807966817500704e36,-1.20370501851145e36)*k*mu*omega**2 + cplx(1.9615933635001409e37,-2.0061750308524168e36)*a*k*mu*omega**2 - cplx(0.,8.024700123409667e35)*a**2*k*mu*omega**2 + 
          cplx(1.8484673825937405e37,4.903983408750352e36)*k**2*mu*omega**2 + 3.5665333881820743e37*a*k**2*mu*omega**2 - cplx(2.0061750308524168e36,1.9615933635001409e37)*a**2*k**2*mu*omega**2 + cplx(4.359096363333646e38,-6.2414334293186295e37)*k**3*mu*omega**2 + 
          cplx(8.718192726667292e38,-7.133066776364149e37)*a*k**3*mu*omega**2 - cplx(0.,3.5665333881820743e37)*a**2*k**3*mu*omega**2 + cplx(3.1207167146593148e37,2.179548181666823e38)*k**4*mu*omega**2 - cplx(7.133066776364149e37,8.718192726667292e38)*a**2*k**4*mu*omega**2 - 
          cplx(0.,3.96281487575786e38)*k**5*mu*omega**2 + 
          cplx(0.,7.92562975151572e38)*a*k**5*mu*omega**2 + 9.90703718939465e37*k**6*mu*omega**2 + 7.92562975151572e38*a**2*k**6*mu*omega**2 - cplx(0.,8.024700123409667e35)*k*mu*ra**2 - cplx(2.006175030852417e36,1.9615933635001409e37)*k**2*mu*ra**2 - cplx(0.,3.5665333881820743e37)*k**3*mu*ra**2 - 
          cplx(7.133066776364149e37,8.718192726667292e38)*k**4*mu*ra**2 + 7.92562975151572e38*k**6*mu*ra**2 + 4.012350061704833e35*k**2*mu**2*ra**2 + 3.5665333881820743e37*k**4*mu**2*ra**2 + 7.92562975151572e38*k**6*mu**2*ra**2 + 4.012350061704833e35*k**2*mu**2*omega**2*ra**2 + 
          3.5665333881820743e37*k**4*mu**2*omega**2*ra**2 + 7.92562975151572e38*k**6*mu**2*omega**2*ra**2 + 0.5*k**2*np.sqrt((mu**2*
               ((k*(cplx(0.,1.6049400246819333e36) + cplx(0.,7.133066776364149e37)*k**2 + k**5*(-1.585125950303144e39 - 1.585125950303144e39*mu) + k**3*(cplx(1.4266133552728297e38,1.7436385453334584e39) - 7.133066776364149e37*mu) + k*(cplx(4.012350061704834e36,3.9231867270002817e37) - 8.024700123409667e35*mu))*ra**2 + 
                    omega**2*(-8.024700123409667e35 - cplx(1.9615933635001409e37,-2.4074100370229e36)*k - cplx(8.718192726667292e38,-1.2482866858637259e38)*k**3 + cplx(0.,7.92562975151572e38)*k**5 + 
                       a**2*k*(cplx(0.,1.6049400246819333e36) + cplx(4.0123500617048336e36,3.9231867270002817e37)*k + cplx(0.,7.133066776364149e37)*k**2 + cplx(1.4266133552728297e38,1.7436385453334584e39)*k**3 - 1.585125950303144e39*k**5) + 
                       a*(-1.6049400246819333e36 - cplx(3.9231867270002817e37,-4.0123500617048336e36)*k - 7.133066776364149e37*k**2 - cplx(1.7436385453334584e39,-1.4266133552728297e38)*k**3 - cplx(0.,1.585125950303144e39)*k**5) + k**6*(-1.98140743787893e38 - 1.585125950303144e39*mu*ra**2) + 
                       k**4*(cplx(-6.2414334293186295e37,-4.359096363333646e38) - 7.133066776364149e37*mu*ra**2) + k**2*(cplx(-3.696934765187481e37,-9.807966817500704e36) - 8.024700123409667e35*mu*ra**2)))**2 - 
                 4.*k**2*(8.024700123409667e35 + 7.133066776364149e37*k**2 + 1.585125950303144e39*k**4)*omega**2*ra**2*(-8.024700123409667e35 + 8.024700123409667e35*mu + 
                    a**2*k*(cplx(0.,-1.6049400246819333e36) - cplx(4.0123500617048336e36,3.9231867270002817e37)*k - cplx(0.,7.133066776364149e37)*k**2 - cplx(1.4266133552728297e38,1.7436385453334584e39)*k**3 + 1.585125950303144e39*k**5)*mu + 
                    k**5*(cplx(0.,-7.92562975151572e38) + mu*(cplx(0.,-7.92562975151572e38) + cplx(0.,1.585125950303144e39)*xa)) + 1.6049400246819333e36*mu*xa + 
                    k*(cplx(-1.9615933635001409e37,2.2067925339376582e36) + mu*(cplx(1.9615933635001409e37,-2.4074100370229e36) - cplx(0.,1.6049400246819333e36)*ra**2 + cplx(3.9231867270002817e37,-4.0123500617048336e36)*xa)) + 
                    k**3*(cplx(-8.718192726667292e38,8.024700123409667e37) + mu*(cplx(8.718192726667292e38,-1.2482866858637259e38) - cplx(0.,7.133066776364149e37)*ra**2 + cplx(1.7436385453334584e39,-1.4266133552728297e38)*xa)) + 
                    a*(2.252599458505449e21 + mu*(1.6049400246819333e36 + cplx(0.,1.585125950303144e39)*k**5 + k**2*(7.133066776364149e37 - cplx(8.024700123409668e36,7.846373454000563e37)*xa) + k*(cplx(3.9231867270002817e37,-4.0123500617048336e36) - cplx(0.,3.2098800493638666e36)*xa) + 
                          k**3*(cplx(1.7436385453334584e39,-1.4266133552728297e38) - cplx(0.,1.4266133552728297e38)*xa) - cplx(2.8532267105456594e38,3.487277090666917e39)*k**4*xa + 3.170251900606288e39*k**6*xa)) + 
                    k**6*(1.98140743787893e38 + mu*(1.98140743787893e38 + 1.585125950303144e39*ra**2) + mu**2*(1.585125950303144e39*ra**2 - 1.585125950303144e39*xa**2)) + 
                    k**4*(cplx(3.5665333881820743e37,2.179548181666823e38) + mu*(cplx(6.2414334293186295e37,4.359096363333646e38) - cplx(1.4266133552728297e38,1.7436385453334584e39)*ra**2) + mu**2*(7.133066776364149e37*ra**2 - 7.133066776364149e37*xa**2)) + 
                    k**2*(cplx(-3.4963172621022395e37,4.903983408750352e36) + mu*(cplx(3.696934765187481e37,9.807966817500704e36) - cplx(4.012350061704834e36,3.9231867270002817e37)*ra**2 + 7.133066776364149e37*xa) + mu**2*(8.024700123409667e35*ra**2 - 8.024700123409667e35*xa**2)))))/k**4))/(k**2*(8.024700123409667e35 + 7.133066776364149e37*k**2 + 1.585125950303144e39*k**4)*mu**2*omega**2*ra**2)
    return val


def compute_flutter(omega=.5, plot=True): 
    ks = np.linspace(5, .5, 10000)

    a = -.3
    xa = .2 
    ra = .3 
    mu = 10 
    e = .2 

    data1 = r1(a, xa, ra, mu, omega, ks)
    data2 = r2(a, xa, ra, mu, omega, ks)
    freq1 = np.sqrt(1/data1.real)
    freq2 = np.sqrt(1/data2.real)

    omega1 = 1/data1.real
    omega2 = 1/data2.real
    g1 = data1.imag/data1.real
    g2 = data2.imag/data2.real

    X = 1/ks

    # print g1[:10] > 0, np.argmax(g1>0)
    # print g2[:10] > 0, np.argmax(g1>0)
    # exit()

    # print g1

    flutter1_idx = np.argwhere(g1 > 1e-10)
    if flutter1_idx.shape[0] > 0: 
        flutter1_idx = flutter1_idx[0,0]
    else: 
        flutter1_idx = 1e10

    flutter2_idx = np.argwhere(g2 > 1e-10)
    if flutter2_idx.shape[0] > 0: 
        flutter2_idx = flutter2_idx[0,0]
    else: 
        flutter2_idx = 1e10

    if flutter1_idx < flutter2_idx:
        f_idx = flutter1_idx
        f_mode = omega1
    elif flutter2_idx < flutter1_idx: 
        f_idx = flutter2_idx
        f_mode = omega2
    else: 
        f_idx = len(X)-1
        f_mode = omega1

    f_k = X[f_idx]
    f_omega = f_mode[f_idx]
    f_speed = f_k*f_omega
    print "flutter freq: ", f_omega
    print "flutter speed: ", f_speed

    if plot: 
        fig, ax = plt.subplots()
        ax.plot(X, omega1)
        ax.plot(X, omega2)

        ax.set_ylim(0,2)
        ax.axvline(f_k, ls="--", c='r')
        ax.set_ylabel(r'$\omega$', fontsize=15)
        ax.set_xlabel(r'$\frac{1}{k}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)

        fig, ax = plt.subplots()
        ax.plot(X, g1)
        ax.plot(X, g2)
        ax.axhline(0, ls="--", c='k')
        ax.axvline(f_k, ls="--", c='r')
        ax.set_ylabel('g', rotation="horizontal", ha="right", fontsize=15)
        ax.set_xlabel(r'$\frac{1}{k}$', fontsize=15)
        ax.set_title(r'$\frac{\omega_h}{\omega_\alpha}=%0.1f$'%omega, va="bottom", fontsize=20)

        plt.show()

    return f_omega, f_speed


compute_flutter(.2, False)
print 
compute_flutter(.5, False)
print 
compute_flutter(.8, False)
