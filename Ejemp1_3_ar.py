#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:05:53 2021

@author: cesartiznado
"""
import numpy as np
import random as rn 
from matplotlib import pyplot as plt
import scipy.signal as sg
# =============================================================================
# 
# =============================================================================

Fm = 200000
dt =  1/(Fm)     #Frecuencia de muestreo 5e-06 (Inverso de 20Khz y 10 veces mas pequeño)
t_ini = 0           #Tiempo de Inicio en seg.
t_end = 0.8         #Tiempo final (seg).
N = np.arange(t_ini,t_end,dt) 
f_0 = 100           # Frecuencia (Hz)
f_1s = 700          #Frecuencia (Hz) 
t_2 = 1.25
Q = 10
f_driver = 200  
rn.seed(8)
n = 160             # Iteraciones descritras por el driven frecuency
t_n = np.random.uniform(t_ini,t_end,n) # uniform distribution of time
t_n = np.sort(t_n)  # Array of the time 
# ================= Padding zeros

tleft = np.arange(-.5,t_ini,dt)
tright = np.arange(t_end,1.5,dt)
t_s = np.arange(t_ini,t_end,dt) #Tiempo en intervalo inicial (simple)
r = np.append(t_s,tright)
t = np.insert(r,0,tleft)

# ================= Variables Declaration of Frequency

f = np.zeros(len(t_s))    # Set memory to get the frequency
p_2 = np.zeros(len(t_s))  # Set memory to p_2
w = np.zeros(len(t_s))    # Set memory to angular frequency

# ================== Set angular frequency and standard frequency

for i in range(0,len(t_s)):
    p_1    =  f_1s - f_0
    p_3    =  t_2 - t_ini
    ab     = (2*t_2-t_ini-1)*(1-t_ini)
    p_2[i] = t_s[i]-t_ini
    f[i] = f_0 + 2*p_1*p_3*(p_2[i])/(ab) - p_1*(p_2[i])**2/(ab)
    w[i] = 2*np.pi*f[i]

a_n1 = np.random.uniform(-1,1,n) 
a_n1 =a_n1 *t_n
# plt.plot(t_n,a_n1,"go")
# plt.title("Uniform Distribution of a_n")
# plt.grid()
# plt.show()

# plt.plot(t_s,f,"r")
# plt.xlim(-.5,1.5,0)
# plt.ylim(0,1000)
# plt.ylabel("f [Hz]")
# plt.xlabel("t [s]")
# plt.title("Frequency of the harmonic oscillator")
# plt.grid()
# plt.show()


# ================= Dirac Delta Implementation

a = np.zeros((len(t_s)))
#s = np.zeros((len(t_s),n))
for i in range(0,len(t_s)):
    for j in range(0,n):
        #s[i,j] =  t_s[i]-t_n[j]
        #a[i] = np.minimum(s[i,:])
        if t_s[i]-t_n[j]<=0 and t_s[i]-t_n[j]>=-15e-6 :     
            #print("-")
            a[i] = a_n1[j]
            i+=1
            
# ================= Sm Euler method

def F(a,h_t,h,w):
    return a - (w/Q)*h_t - (w**2)*h

hs_t = np.zeros(len(t_s))
hs = np.zeros(len(t_s))

for i in range(0,len(t_s)):
    hs_t[i] = hs_t[i-1] + F(a[i-1],hs_t[i-1],hs[i-1],w[i-1])*dt
    hs[i] = hs[i-1] + hs_t[i-1]*dt

# # ================ plot
hrms = 1e-9
hs = hs/hrms            
# plt.plot(t_s,hs,"black")
# plt.title("Semi-Euler")
# plt.xlim(-0.5,1.0)
# plt.xlabel("t [s]")
# plt.ylabel("h/$h_{rms}$")
# plt.grid()
# plt.show()
# Add zeros to the angular frequeny,frequency, the signal
hsr = np.zeros(len(tright))
hsl = np.zeros(len(tleft))
r3  = np.append(hs,hsr)
hss  = np.insert(r3,0,hsl)

wr = np.zeros(len(tright))
wl = np.zeros(len(tleft))
r1 = np.append(w,wr)
w  = np.insert(r1,0,wl) 

fr = np.zeros(len(tright))
fl = np.zeros(len(tleft))
r2 = np.append(f,fr)
f  = np.insert(r2,0,fl)

# plt.plot(t,hss,"black")
# plt.title("Semi-Euler")
# #plt.xlim(-0.5,1.0)
# plt.xlabel("t [s]")
# plt.ylabel("h/$h_{rms}$")
# plt.grid()
# plt.show()

# =============================================================================
# Resample to (16386) Hz 
# =============================================================================
from scipy import interpolate
sample = 16386
def sn_resample_wave(t,h,fs):
    """
    Interpolate array h to the fs sampling frequency.
   
    Input:
        t  - time array, in seconds
        h  - strain array to be interpolated
        fs - thi is the new sampling frequency
    Output:
        t1 - time array, after resampling
        h1 - new strain array
    """
   
    # Quick check
    if len(t)!=len(h):
        print("Error: t and h need to have equal sizes")
        return 0
   
    # Define new time with fs
    t1 = np.arange(t[0],t[-1],1.0/fs)
   
    # Interpolation
    tck = interpolate.splrep(t,h,s=0)
    h1  = interpolate.splev(t1,tck,der=0)
   
    return t1, h1


t1,h1 = sn_resample_wave(t,hss,sample)
plt.plot(t1,h1,"black")
plt.title("Resample")
plt.xlim(-0.5,1.0)
plt.xlabel("t [s]")
plt.ylabel("h/$h_{rms}$")
plt.grid()
plt.show()

# =============================================================================
# Peridiogram
# =============================================================================




wnd="hamming"
nfft=4*4096
nperseg=4096
noverlap=4096-512
fxx, txx, Sxx = sg.spectrogram(hs, Fm, window=wnd, nfft=nfft, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
# Problemas con la funcion txx en la region -.5 a 0 
plt.figure()
plt.pcolormesh(txx, fxx, Sxx,shading="auto")
plt.plot(t,f,"r")
plt.title("Frequency of the harmonic oscillator (Simple)")
cbar = plt.colorbar()
cbar.set_label('Amplitude strain [1/Hz]')
plt.ylim([0,1000])
#plt.title(r"{0:s} hplus  $(\phi,\theta)=({1:.0f}, {2:.0f})$".format(name, phi_eq*180.0/np.pi, theta_eq*180.0/np.pi))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()


# ======================= 
fxx, txx, Sxx = sg.spectrogram(hss, Fm, window=wnd, nfft=nfft, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
# Problemas con la funcion txx en la region -.5 a 0 
plt.figure()
plt.pcolormesh(txx, fxx, Sxx,shading="auto")
plt.plot(t,f,"r")
plt.title("Frequency of the harmonic oscillator (Simple)")
cbar = plt.colorbar()
cbar.set_label('Amplitude strain [1/Hz]')
plt.ylim([0,1000])
#plt.title(r"{0:s} hplus  $(\phi,\theta)=({1:.0f}, {2:.0f})$".format(name, phi_eq*180.0/np.pi, theta_eq*180.0/np.pi))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()


# dtspec = txx[1]-txx[0]
# ========================== Padding zeros to time
txl = np.arange(-.5,t_ini,.1)
txr = np.arange(t_end,1.5,.1)
r4  = np.append(txx,txr)
txxz = np.insert(r4,0,txl)
# ========================== Padding zeros to frequency
fxl = np.zeros(len(txl))
fxr = np.zeros(len(txr))
r5  = np.append(fxx,fxr)
fxxz = np.insert(r5,0,fxl)
# ========================== Padding the length of the previous zeros to the signal
sxx = np.zeros((len(fxxz),len(txxz)))
for i in range(0, len(fxx)-1):
    for j in range(0,304):
        sxx[i + len(txl),j + len(fxl) ] = Sxx[i,j]
        #print(i,j)
# ========================== Logaritmic scale to Sxx
    """
    Found a problem when we get the logaritmical scale of the signal hs,
    so i makeit to the Sxx signal.
    """
#Sxx = np.log(Sxx)
plt.figure()
plt.pcolormesh(txxz, fxxz, sxx,shading="auto")
#plt.plot(t,f,"w")
plt.title("Frequency of the harmonic oscillator")
#plt.plot(t,f,"r")
cbar = plt.colorbar()
cbar.set_label('Amplitude strain [1/Hz]')
plt.ylim([0,1000])
#plt.title(r"{0:s} hplus  $(\phi,\theta)=({1:.0f}, {2:.0f})$".format(name, phi_eq*180.0/np.pi, theta_eq*180.0/np.pi))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()
        
        
        
        
        
# =========================== Intento Fallido de Sxx
# Sxt = np.zeros((len(Sxx[:]), len(txl)))
# Sxf = np.zeros((len(txl),len(Sxx[:])))
# r6  = np.insert(Sxx,len(Sxx[0]),Sxf,axis=1)
# r7  = np.append(r6,Sxt, axis=0)
#Sxx = np.insert(Sxx,[0,len(Sxx[:])],Sxf)

