#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:17:57 2021

@author: cesartiznado
"""
# =============================================================================
# Generación de Ondas 
# =============================================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rn
from matplotlib import pyplot as plt
import simpy as sp
# =============================================================================
# Declaración de variables
# =============================================================================
N = 160             #Numero de Iteraciones distinto al de delta
t_ini = 0           #Tiempo de Inicio en seg.
t_end = 0.8         #Tiempo final (seg).
f_0 = 100           # Frecuencia (Hz)
f_1s = 700          #Frecuencia (Hz) 
t_2 = 1.25
Q = 10
f_driver = 200      
t = np.linspace(t_ini,t_end,N) #Tiempo sobre el cual trabajamos

dt = np.zeros(N)
for i in range(0,N): #1/4096*4 = Frecuencia de muestreo, dt = 1/F_s
    dt[i] = t[i] - t[i-1] # diferencial del tiempo


f = np.zeros(N,dtype=float)
p_1 = np.zeros(N)
p_2 = np.zeros(N)
t = np.linspace(t_ini,t_end,N)
w = np.zeros(N)
h = np.zeros(N)
h_v = np.zeros(N)


# =============================================================================
# Definir nuestra frecuencia y frecuencia angular
# =============================================================================
for i in range(0,N):
    p_1    =  f_1s - f_0 
    p_2[i] = t[i]-t_ini
    p_3    =  t_2 - t_ini
    ab = (2*t_2-t_ini-1)*(1-t_ini)
    f[i] = f_0 + 2*p_1*p_3*(p_2[i])/(ab) - p_1*(p_2[i])**2/(ab)
    w[i] = 2*np.pi*f[i]
# =============================================================================
# Definir distribución aleatoria de la amplitud y el tiempo 
# Para la aceleración  (Esta N esta bien, corresponde f_drive.)
# =============================================================================
a_mx = w[N-1]**2                #Amplitud maxima
Amp = np.linspace(0,a_mx,N)     #Amplitud
tran = rn.choices(t,k=N)        #Distribucion aleatoria del tiempo.
Amp_ran = rn.choices(Amp,k=N)   #Distribución aleatoria de la amplitud.

c = np.zeros(N)
c1 = np.zeros(N)
for i in range(0,N-1):
    c[i] = 1/ ( 1/(dt[i]**2) + w[i]/(Q*dt[i]))
    c1[i] = ( w[i]**2 - w[i]/(Q*dt[i]) - 2/(dt[i]**2) )
    h[i+1] = (Amp_ran[i] - h[i-1]/(dt[i])**2 - h[i]*c1[i])*c[i]
    print(i)



# dlt = sp.series.limit()



gra = Amp_ran/a_mx
#hs = h/h_rs 


plt.plot(gra,t,"ro")
plt.xlabel("t (ms)")
plt.ylabel("$a_{n}/a_{max}$")


# =============================================================================
# Attempt #2
# =============================================================================

# s=np.zeros(N)
# for i in range(0,N):
#     s[i] = t[i]-tran[i]
# def f(h,h_v):
#     return -(w*h_v)/Q - (w**2)*h


# h_v = np.zeros(N)
# h = np.zeros(N)

# for i in range(0,N):
#     h_v[i+1] = h_v[i] + f(h,h_v)*dt    
#     h[i+1] = h[i] + h_v*dt


