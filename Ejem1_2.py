#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:49:21 2021

@author: cesartiznado
"""
import numpy as np
import random as rn 
from matplotlib import pyplot as plt
import scipy as sci 
# ===================================== Declaración de variables
Fm = dt =  1/(200000)     #Frecuencia de muestreo 5e-06 (Inverso de 20Khz y 10 veces mas pequeño)
#dt = Fm
t_ini = 0           #Tiempo de Inicio en seg.
t_end = 0.8         #Tiempo final (seg).
N = np.arange(t_ini,t_end,dt)     #Numero de Iteraciones distinto al de delta
f_0 = 100           # Frecuencia (Hz)
f_1s = 700          #Frecuencia (Hz) 
t_2 = 1.25
Q = 10
f_driver = 200      
t = np.linspace(t_ini,t_end,len(N)) #Tiempo sobre el cual trabajamos
#rn.seed(0)                       #Definimos una semilla de numeros aleatorios
n = 160                          #Iteraciones descritras por el driven frecuency
t_n = np.random.uniform(t_ini,t_end,160)
t_n = np.sort(t_n)
# =============================================================================
# =============================================================================
#                           Modelo de Oscilador Armonico 
# =============================================================================
# =============================================================================
#                           Declaración de Variables para nuestro Modelo
# =============================================================================
f = np.zeros(len(N))
p_1 = np.zeros(len(N))
p_2 = np.zeros(len(N))
#t = np.linspace(t_ini,t_end,N)
w = np.zeros(len(N))
h = np.zeros(len(N))
h_v = np.zeros(len(N))
# =============================================================================
# Definir nuestra frecuencia y frecuencia angular
# =============================================================================
rn.seed(8)
for i in range(0,len(N)):
    p_1    =  f_1s - f_0 
    p_2[i] = t[i]-t_ini
    p_3    =  t_2 - t_ini
    ab = (2*t_2-t_ini-1)*(1-t_ini)
    f[i] = f_0 + 2*p_1*p_3*(p_2[i])/(ab) - p_1*(p_2[i])**2/(ab)
    w[i] = 2*np.pi*f[i]

a_n1 = np.random.uniform(-1,1,160)
a_n1 =a_n1 *t_n
plt.plot(t_n,a_n1,"go")
plt.title("Uniform Distribution of a_n")
plt.grid()
plt.show()
# =============================================================================
# Metodo de Euler para determinar perturbación h
# =============================================================================
# ====================================== Finite differences
a = np.zeros((len(t)))
s = np.zeros((len(t),n))
for i in range(0,len(t)):
    for j in range(0,n):
       s[i,j] =  t[i]-t_n[j]
       #a[i] = np.minimum(s[i,:])
       if t[i]-t_n[j]<=0 and t[i]-t_n[j]>=-15e-6 :          #t[i]-t_n[j]<=0
           #print("-")
           a[i] = a_n1[j]
           i+=1

hrms = 1e-9     # El factor

c = np.zeros(len(N))
c1 = np.zeros(len(N))
for i in range(0,len(N)-1):
    c[i] = 1/ ( 1/(dt**2) + w[i]/(Q*dt))
    c1[i] = ( w[i]**2 - w[i]/(Q*dt) - 2/(dt**2) )
    h[i+1] = (a[i] - h[i-1]/(dt)**2 - h[i]*c1[i])*c[i]
# h = h/hrms
plt.plot(t,h,"black")
plt.title("Finite Differences")
plt.xlim(-0.5,1.0)
#plt.ylim(-8,8)
plt.xlabel("t [s]")
plt.ylabel("h/$h_{rms}$")
plt.grid()
plt.show()
# ======================================= Semi-implicit Euler method
def F(a,h_t,h,w):
    return a - (w/Q)*h_t - (w**2)*h

hs_t = np.zeros(len(N)+1)
hs = np.zeros(len(N)+1)
t = np.linspace(t_ini,t_end,len(N)+1)
# hs_t[0] = 0
# hs[0] = 0 
for i in range(0,len(N)):
    hs_t[i+1] = hs_t[i] + F(a[i],hs_t[i],hs[i],w[i])*dt
    hs[i+1] = hs[i] + hs_t[i]*dt
    
hs = hs/hrms            #Normalizacion?
plt.plot(t,hs,"black")
plt.title("Semi-Euler")
plt.xlim(-0.5,1.0)
#plt.ylim(-8,8)
plt.xlabel("t [s]")
plt.ylabel("h/$h_{rms}$")
plt.grid()
plt.show()

# =============================================================================
# PSD
# =============================================================================
PSD = sci.signal.spectrogram(hs)