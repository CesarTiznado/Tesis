#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:13:14 2021

@author: cesartiznado
"""
import numpy as np
import random as rn 
from matplotlib import pyplot as plt
# =============================================================================
# 
# =============================================================================
Fm = 1/(4096*4)     #Frecuencia de muestreo
dt = Fm
t_ini = 0           #Tiempo de Inicio en seg.
t_end = 0.8         #Tiempo final (seg).
N = np.arange(t_ini,t_end,dt)     #Numero de Iteraciones distinto al de delta
f_0 = 100           # Frecuencia (Hz)
f_1s = 700          #Frecuencia (Hz) 
t_2 = 1.25
Q = 10
f_driver = 200      
t = np.linspace(t_ini,t_end,len(N)) #Tiempo sobre el cual trabajamos


# =============================================================================
# a(t) = a_n*delta
# =============================================================================

n = 160                          #Iteraciones descritras por el driven frecuency
t_n = np.linspace(t_ini,t_end,n) #Tiempo de comparativa
a_max = 20                       #Amplitud maxima
a_n = np.linspace(0,a_max,n)     #Amplitud de onda (h)
rn.seed(0)                       #Definimos una semilla de numeros aleatorios
a_nran = rn.choices(a_n,k=len(a_n)) #Distribucion aleatoria de amplitudes
t_nran = rn.choices(t_n,k=len(t_n)) #Distribucion aleatoria de tiempos permitidos
#rn.shuffle(a_n)
#a_n1 = rn.shuffle(a_n)
a = np.zeros((len(t)))
s = np.zeros((len(t),n))
for i in range(0,len(t)):
    for j in range(0,n):
       s[i,j] =  t[i]-t_nran[j]
       #a[i] = np.minimum(s[i,:])
       if t[i]-t_n[j]<=0 and t[i]-t_n[j]>=-15e-6 :          #t[i]-t_n[j]<=0
           #print("-")
           a[i] = a_nran[j]
           i+=1
           
r = np.zeros(len(t_nran))           
for i in range(0,len(t_nran)):
    r[i] = a_nran[i]/a_max
#plt.plot(t,a,"bo")    
#plt.plot(t_n,r,"ro")           #Grafica del presunto a_n/a_max
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
for i in range(0,len(N)):
    p_1    =  f_1s - f_0 
    p_2[i] = t[i]-t_ini
    p_3    =  t_2 - t_ini
    ab = (2*t_2-t_ini-1)*(1-t_ini)
    f[i] = f_0 + 2*p_1*p_3*(p_2[i])/(ab) - p_1*(p_2[i])**2/(ab)
    w[i] = 2*np.pi*f[i]
# =============================================================================
# Metodo de Euler para determinar perturbación h
# =============================================================================

c = np.zeros(len(N))
c1 = np.zeros(len(N))
for i in range(0,len(N)-1):
    c[i] = 1/ ( 1/(dt**2) + w[i]/(Q*dt))
    c1[i] = ( w[i]**2 - w[i]/(Q*dt) - 2/(dt**2) )
    h[i+1] = (a[i] - h[i-1]/(dt)**2 - h[i]*c1[i])*c[i]
    #print(i)

# =============================================================================
# Parte II Metodo euler
# =============================================================================
plt.plot(t,h,"db")
plt.xlabel("t (s)")
plt.ylabel("h")








# =============================================================================
# Distribucion gaussiana
# =============================================================================
# l = np.random.normal(0,1.0,a_n)
