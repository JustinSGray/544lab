import numpy as np
import matplotlib.pylab as plt

cplx = complex


def r1(a, xa, ra, mu, omega, u):
    val = ((-0.5*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa))/
          (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
     -   0.5*np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) + (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)) - 
     -   0.5*np.sqrt((2.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2)/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (1.3333333333333333*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) - (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.25*((-8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**3)/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**3 + 
                 (8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa))/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2\
                  - (64.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u))/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
               (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                    8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
               (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                     (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                       16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                    96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                     (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
                ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
                  (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333) + 
               (0.26456684199469993*(864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333)/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))))
    return val 


def r2(a, xa, ra, mu, omega, u):
    val = ((-0.5*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa))/
          (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
     -   0.5*np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) + (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)) + 
     -   0.5*np.sqrt((2.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2)/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (1.3333333333333333*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) - (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.25*((-8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**3)/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**3 + 
                 (8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa))/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2\
                  - (64.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u))/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
               (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                    8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
               (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                     (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                       16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                    96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                     (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
                ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
                  (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333) + 
               (0.26456684199469993*(864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333)/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))))
    return val 


def r3(a, xa, ra, mu, omega, u):
    val = ((-0.5*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa))/
          (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
     -   0.5*np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) + (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)) - 
     -   0.5*np.sqrt((2.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2)/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (1.3333333333333333*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) - (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.25*((-8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**3)/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**3 + 
                 (8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa))/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2\
                  - (64.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u))/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
               (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                    8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
               (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                     (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                       16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                    96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                     (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
                ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
                  (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333) + 
               (0.26456684199469993*(864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333)/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))))
    return val 


def r4(a, xa, ra, mu, omega, u):
    val = ((-0.5*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa))/
          (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
     -   0.5*np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) + (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)) + 
     -   0.5*np.sqrt((2.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2)/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
            (1.3333333333333333*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                 8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
             (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
            (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                  (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                 (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                    16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333) - (0.26456684199469993*
               (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                  144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                   (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa) + 2.*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                  1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                   (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                  576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                   (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                     16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                  np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                         (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                        96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                         (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                    (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                       144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                        (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                       2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                           16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                       1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                        (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                       576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                        (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                          16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))\
                      **2))**0.3333333333333333)/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
            (0.25*((-8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**3)/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**3 + 
                 (8.*(u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                      16.*a*mu*u**2 - 16.*mu*u**2*xa))/(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2\
                  - (64.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u))/
                  (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
             np.sqrt((u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)**2 - 
               (0.6666666666666666*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                    8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa))/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
               (0.41997368329829105*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                     (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                    (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                       16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                    96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                     (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)))/
                ((1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2)*
                  (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333) + 
               (0.26456684199469993*(864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                     144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                      (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                     2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                         16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                     1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                      (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                     576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                      (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                        16.*a*mu*u**2 - 16.*mu*u**2*xa)*(1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) + 
                     np.sqrt(-4.*(-48.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                            (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa) + 
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                              16.*a*mu*u**2 - 16.*mu*u**2*xa)**2 + 
                           96.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                            (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**3 + 
                       (864.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)**2 - 
                          144.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)*
                           (u - 4.*a*mu*u + 8.*a**2*mu*u + 8.*mu*ra**2*u - 4.*mu*u*xa + 16.*a*mu*u*xa)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa) + 
                          2.*(mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 
                              8.*mu*u**2 - 16.*a*mu*u**2 - 16.*mu*u**2*xa)**3 + 
                          1728.*(-1.*a*mu*omega**2*u + 2.*a**2*mu*omega**2*u + 2.*mu*ra**2*u)**2*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2) - 
                          576.*(mu**2*omega**2*ra**2 - 1.*mu*omega**2*u**2 - 2.*a*mu*omega**2*u**2)*
                           (mu*omega**2 + 8.*a**2*mu*omega**2 + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 8.*mu**2*omega**2*ra**2 + 8.*u**2 - 8.*mu*u**2 - 
                             16.*a*mu*u**2 - 16.*mu*u**2*xa)*
                           (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))**2))**0.3333333333333333)/
                (1. + mu + 8.*a**2*mu + 8.*mu*ra**2 + 8.*mu**2*ra**2 + 16.*a*mu*xa - 8.*mu**2*xa**2))))
    return val 


def compute_flutter(omega, plot=True): 
    a = -.3
    xa = .2 
    ra = .3 
    mu = 10 
    e = .2 

    U = np.linspace(1e-10,2,300, dtype="complex")
    data1 = r1(a, xa, ra, mu, omega, U)
    data2 = r2(a, xa, ra, mu, omega, U)
    data3 = r3(a, xa, ra, mu, omega, U)
    data4 = r4(a, xa, ra, mu, omega, U)

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

        fig, ax = plt.subplots()
        ms = 10
        ax.scatter(U.real, data1.imag, c=colors[0], s=ms, lw=0)
        ax.scatter(U.real, data2.imag, c=colors[1], s=ms, lw=0)
        ax.scatter(U.real, data3.imag, c=colors[2], s=ms, lw=0)
        ax.scatter(U.real, data4.imag, c=colors[3], s=ms, lw=0)
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
