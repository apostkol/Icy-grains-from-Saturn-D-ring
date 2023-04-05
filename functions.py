import numpy as np
import scipy
import sympy as sp
from sympy import *
from scipy.optimize import fsolve
from scipy.optimize import bisect
from matplotlib import pyplot as plt
NA=6.0221408e+23

def gravity(G, Ms, state):
    
    r=state[0][0][0]
    a_grav = -G*Ms/r**2
    
    return np.array([a_grav, 0, 0])

def rho_a(state, n_half=8.9e22, a1=1/219, r_half=2340, Rs=60268):

    r=state[0][0][0]-Rs
    rho=n_half*np.exp(-a1*(r-r_half))

    return rho

def drag(Gamma, rho_m, rho_a, R_grain, state):
    
    pos=state[0][0]; pos_der=state[1][0]
    r=pos[0]; theta=pos[1]; phi=pos[2]
    v_r=pos_der[0]; v_theta=r*pos_der[1]; v_phi=r*np.sin(theta)*pos_der[2]
    v=np.array([v_r, v_theta, v_phi])
    A = 3*rho_a/(4*rho_m*R_grain)
    v_mag=np.linalg.norm(v)
    a_drag = -A*Gamma*v_mag*v*np.array([1, 1/r, 1/(r*np.sin(theta))])
    #print(a_drag)
    return a_drag

def drag2(Gamma, rho_m, rho_a, R_grain, state):
    pos=state[0][0]; pos_der=state[1][0]
    r=pos[0]; theta=pos[1]; phi=pos[2]
    v_r=pos_der[0]; v_theta=r*pos_der[1]; v_phi=r*np.sin(theta)*pos_der[2]
    
    v=np.array([v_r, v_theta, v_phi])
    A = 3*rho_a/(4*rho_m*R_grain)
    v_mag=np.linalg.norm(v)
    a_drag = -A*Gamma*v_mag*v
    #print(a_drag)
    return a_drag

def acc_RHS(state):
    
    pos=state[0][0]; pos_der=state[1][0]
    r=pos[0]; theta=pos[1]; phi=pos[2]
    v_r=pos_der[0]; v_theta=r*pos_der[1]; v_phi=r*np.sin(theta)*pos_der[2]
    
    RHS= [v_theta**2/r + v_phi**2/r ,
          v_phi**2/r**2 * np.cos(theta)/np.sin(theta) - 2*v_r*v_theta/r**2,
          -2*v_r*v_phi/(r**2*np.sin(theta)) - 2*v_theta*v_phi/r**2 *np.cos(theta)/np.sin(theta)**2]
    
    return np.array(RHS)

def vel_RHS(state):
    
    pos_der=state[1][0]
    
    vel_RHS=[pos_der[0],
             pos_der[1],
             pos_der[2]]
    
    return np.array(vel_RHS)

def e(T):
    e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
    
    return e

def L(T):
    L=1000*(2834.1-0.29*T-0.004*T**2)
    
    return L
    
def temp(v, rho_a, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23):
   
    def func(T, v, rho_a, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23):
        e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
        L=1000*(2834.1-0.29*T-0.004*T**2)
        
        return v**3*rho_a-8*ep*sigma/lam *(T**4-Tenv**4+e*L/(ep*sigma) * np.sqrt((mu/NA)/(2*np.pi*kb*T)))
    
    T_guess=150
    T_sol=fsolve(func, T_guess, args=(v, rho_a))
    
    return T_sol

def mass(rho, T, m0, i, dt, mu=18.01528e-3, kb=1.380649e-23, NA=6.0221408e+23):
  
    e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
    
    #m=(-(4*np.pi)**(1/3)*(3/rho)**(2/3)/3 *e*np.sqrt((mu/NA)/(2*np.pi*kb*T))*(i-1)*dt+m0**(1/3))**3
    
    m=m0-4*np.pi*(3*m0/(4*np.pi*rho))**(2/3)*e*np.sqrt((mu/NA)/(2*np.pi*kb*T))*dt

    return m    