import numpy as np
import scipy
import sympy as sp
from sympy import *
from scipy.optimize import fsolve
from scipy.optimize import bisect
from matplotlib import pyplot as plt
from sympy import symbols, nonlinsolve
from sympy import log, exp, sqrt

def gravity(pos, pos_der, G, Ms):
    x=pos[0]; y=pos[1]; z=pos[2]
    r=np.sqrt(x**2+y**2+z**2)
    a_grav = -G*Ms/r**2
    
    theta=np.arccos(z/r)
    phi=np.arctan2(y,x)
    
    return a_grav*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def rho_a(pos, pos_der, n_half=8.9e13, a1=1/(219*1e3), r_half=2340*1e3, Rs=60268*1e3):
    x=pos[0]; y=pos[1]; z=pos[2]
    r=np.sqrt(x**2+y**2+z**2)-Rs
    rho=n_half*np.exp(-a1*(r-r_half))

    return rho

def drag(pos, pos_der, R_grain, rho_m, Gamma, M_H2=2.016e-3, M_ice=18.02e-3, NA=6.0221408e+23):
    x=pos[0]; y=pos[1]; z=pos[2]
    vx=pos_der[0]; vy=pos_der[1]; vz=pos_der[2]
    v=np.array([vx, vy, vz])
    A = 3*rho_a(pos, pos_der)*(M_H2/NA)/(4*rho_m*(M_ice/NA)*R_grain)
    v_mag=np.linalg.norm(v)
    a_drag_x=-A*Gamma*v_mag*vx; a_drag_y=-A*Gamma*v_mag*vy; a_drag_z=-A*Gamma*v_mag*vz
    #a_drag = -A*Gamma*v_mag**2*v/v_mag
    a_drag=np.array([a_drag_x, a_drag_y, a_drag_z])
    
    return a_drag

def magnetic(pos, pos_der, Q, m, Rs, Mom, Om):
    
    x=pos[0]; y=pos[1]; z=pos[2]
    vx=pos_der[0]; vy=pos_der[1]; vz=pos_der[2]
    
    z_s = 0.04*Rs
    r_s = np.sqrt(x**2+y**2+(z-z_s)**2)
    
    Bx = 3*Mom*x*(z-z_s)/r_s**5  
    By = 3*Mom*y*(z-z_s)/r_s**5  
    Bz = Mom*(3*(z-z_s)**2-r_s**2)/r_s**5
        
    B = np.array([Bx, By, Bz]) #B in cartesian
    v = np.array([vx, vy, vz])
    
    #phi=np.arctan2(y,x)
   
    Om_vector=np.array([0,0,Om])
    Om_x_r = np.cross(Om_vector, pos) #Ωxr
    
    E_c = -Q/m * np.cross(Om_x_r, B) #-Q/m*(Ωxr)xB
    
    VxB = Q/m * np.cross(v, B) #Q/m*VxB
    
    return E_c, VxB

def vel_RHS(pos, pos_der):
        
    vel_RHS=[pos_der[0],
             pos_der[1],
             pos_der[2]]
    
    return np.array(vel_RHS)
    
def temp(pos, pos_der, v, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23, M_H2=2.016e-3):
    rho_atm=rho_a(pos, pos_der)*M_H2/NA
    def func(T, v, rho_atm, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23):
        e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
        L=1000*(2834.1-0.29*T-0.004*T**2)
        
        return v**3*rho_atm-8*ep*sigma/lam *(T**4-Tenv**4+e*L/(ep*sigma) * np.sqrt((mu)/(2*np.pi*kb*NA*T)))
    
    T_guess=90
    T_sol=fsolve(func, T_guess, args=(v, rho_atm), xtol=1e-5, maxfev=5000)
    
    return T_sol

def mass(T, m0, t, dt, rho_m_mass, mu=18.01528e-3, kb=1.380649e-23, NA=6.0221408e+23):
    e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
    
    m=(-(4*np.pi)**(1/3)*(3/rho_m_mass)**(2/3)/3 *e*np.sqrt((mu)/(2*np.pi*kb*NA*T))*t+m0**(1/3))**3
    
    #m = m0 - 4*np.pi*(3*m0/(4*np.pi*rho_m_mass))**(2/3)*e*np.sqrt((mu/NA)/(2*np.pi*kb*T))*dt

    return m    

def temp2(pos, pos_der, v, T_lim, n, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23, M_H2=2.016e-3):
    
    rho_atm=rho_a(pos, pos_der)*M_H2/NA
    
    def f(T, v, rho_atm, ep=1, sigma=5.670374e-8, lam=1, mu=18.01528e-3, kb=1.380649e-23, Tenv=85, NA=6.0221408e+23):
        T=np.array(T)
        e=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
        L=1000*(2834.1-0.29*T-0.004*T**2)
        
        return v**3*rho_atm-8*ep*sigma/lam *(T**4-Tenv**4+e*L/(ep*sigma) * np.sqrt((mu)/(2*np.pi*kb*NA*T)))
    
    T_vec = np.linspace(T_lim[0], T_lim[1], n)
    y = f(T_vec, v, rho_atm)
    roots = []
    for i in range(n-1):
        if y[i]*y[i+1] < 0:
            root = T_vec[i] - (T_vec[i+1] - T_vec[i])/(y[i+1] - y[i])*y[i]
            roots.append(root)
    
    return roots

