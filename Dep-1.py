#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 23:12:07 2021

@author: cesartiznado
"""
import numpy as np
import random as rn 
from matplotlib import pyplot as plt
import scipy.signal as sg
import pandas as pd
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


# =============================================================================


# =================== Inicial variables

def cond(Fm,t_ini,t_end,f_0,f_1,t_2,f_driver,Q,seed,OG):
    """
    Fm: Frecuencia de muestreo, t_ini:Tiempo inicial, t_end:Tiempo final
    f_0,f_1,t_2,f_driver,Q,ran_seed
    """
    dt = 1/Fm
    N  = np.arange(t_ini,t_end,dt) 
    n  = int(f_driver/(t_end-t_ini))
    t_n = np.random.uniform(t_ini,t_end,n) #Set the uniform distribution of t_n
    t_n = np.sort(t_n)       # t_n in progresive order
    t_s = np.arange(t_ini,t_end,dt) #Time in the inicial interval (simple)
    rn.seed(seed)
    f = np.zeros(len(t_s))    # Set memory to get the frequency
    p_2 = np.zeros(len(t_s))  # Set memory to p_2
    w = np.zeros(len(t_s))    # Set memory to angular frequency
    
    for i in range(0,len(t_s)):
        p_1    =  f_1 - f_0
        p_3    =  t_2 - t_ini
        ab     = (2*t_2-t_ini-1)*(1-t_ini)
        p_2[i] = t_s[i]-t_ini
        f[i] = f_0 + 2*p_1*p_3*(p_2[i])/(ab) - p_1*(p_2[i])**2/(ab)
        w[i] = 2*np.pi*f[i]
    
    return Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG


# =================== Delta implementation

def delta_imp():
    a_n1 = np.random.uniform(-1,1,n) 
    a_n1 =a_n1 *t_n
    a = np.zeros((len(t_s)))
    for i in range(0,len(t_s)):
        for j in range(0,n):
            #s[i,j] =  t_s[i]-t_n[j]
            #a[i] = np.minimum(s[i,:])
            if t_s[i]-t_n[j]<=0 and t_s[i]-t_n[j]>=-15e-6 :     
                #print("-")
                a[i] = a_n1[j]
                i+=1
    return a,a_n1

# ============== Model Implementation

def model(): 
    h_t = np.zeros(len(t_s))
    h = np.zeros(len(t_s))
    def F(a,h_t,h,w):
        return a - (w/Q)*h_t - (w**2)*h
    for i in range(0,len(t_s)):
        h_t[i] = h_t[i-1] + F(a[i-1],h_t[i-1],h[i-1],w[i-1])*dt
        h[i] = h[i-1] + h_t[i-1]*dt
    hrms = 1e13
    h = h/hrms
    h = h/2
    #h = h/max(abs(h))
    return h


# ========================== Resample to 16386

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

# ================ Add zeros to the angular frequeny,frequency, the signal and time
def pad_zeros():
    tright = np.arange(t_end,1.0,dt)
    tz = np.append(t_s,tright)
        
    hsr = np.zeros(len(tright))
    hz  = np.append(h,hsr)
    
    wr = np.zeros(len(tright))
    wz = np.append(w,wr)
    
    fr = np.zeros(len(tright))
    fz = np.append(f,fr)
    return tz,wz,fz,hz

       
def save(datog):
    datog = pd.DataFrame()
    datog["Time"] = tz
    datog["Frequency"] = fz
    datog["Signal"] = hz
    # datog["SG_Time"] = t
    # datog["SG_Strain"] = hx_SG
    return datog.to_csv("Datog"+str(OG)+".csv",header=True,index = False)


wnd="hamming"
nfft=4*4096
nperseg=4096
noverlap=4096-512

# =============================================================================
# Inicial Conditions of our Perturbartion
# =============================================================================
# =============== OG_1
#Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG = cond(200000,0.0,0.5,100,1100,.6,600,10,8,1)

# =============== OG_2
#Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG = cond(200000,0.0,0.2,100,400,.4,100,1,8,2)

# =============== OG_3
Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG = cond(200000,0.0,0.5,100,700,1.25,200,10,8,3)

# =============== OG_4

#Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG = cond(200000,0.2,0.9,300,1400,1.2,400,10,8,4)

# =============== OG_5
#Fm,N,n,t_s,t_n,f,w,Q,dt,t_ini,t_end,OG = cond(200000,0.1,0.6,150,800,.8,250,1,8,5)


# =============================================================================
# Call de functions to execute the instructions
# =============================================================================

a,a_n1  = delta_imp()                  #Dirac Delta
h = model()                            #Strain
tz,wz,fz,hz = pad_zeros()              #Paddig zeros to the signal 
t1,h1 = sn_resample_wave(tz,hz,sample) #Resample to LIGO frequency
#hp = wfSG(sample)                      #hp polarization in Sine Gaussian waveforms

# ===================== Spectrogram
fxz, txz, Sxz = sg.spectrogram(hz, Fm, window=wnd, nfft=nfft, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
txz = txz + tz[0] # Recorrer el espectrograma
#hp,hx,hp_GS,hx_GS,exp = wfG()
# =============================================================================
# 
# =============================================================================

import h5py
from sn_library import * 


cm2kpc = 3.24078e-22       # Constants
kpc2m  = 3.08567758128e+19 # m
D10kpc = 10.0 * kpc2m # 10 kpc in m

c = 2.99792458e8 # m/s
G = 6.67430e-11 # m^3 kg^−1 s^−2
msun = 1.9885e+30 # kg
esun = msun*c**2 # J

def sn_butfilt_wave(h,fcri,fsam):
    """
    High-pass Butterworth digital filter, to remove low frequencies in wave time series
    
    Input:
        h    - strain array
        fcri - critical frequency
        fsam - sampling frequency
    Output:
        h1   - new strain array
    """
    
    # Define the high-pass filter
    sos = signal.butter(5, fcri, 'highpass', fs=fsam, output='sos')
    
    # Apply the high-pass filter
    h1 = signal.sosfilt(sos, h)
    
    return h1

#-------------------------------> Preliminary version, I have to confirm this
def sn_butfilt_quad(qij,fcri,fsam):
    """
    High-pass Butterworth digital filter to remove low frequencies in quadrupole matrix
    
    Input:
        qij  - input N x 3 x 3 quadrupole moment array
        fcri - critical frequency
        fsam - sampling frequency
    Output:
        qij - output N x 3 x 3 quadrupole moment array
    """
    
    # Define the high-pass filter
    sos = signal.butter(5, fcri, 'highpass', fs=fsam, output='sos')
    
    # Apply the high-pass filter
    qij[:,0,0] = signal.sosfilt(sos, qij[:,0,0])
    qij[:,0,1] = signal.sosfilt(sos, qij[:,0,1])
    qij[:,0,2] = signal.sosfilt(sos, qij[:,0,2])
    qij[:,1,1] = signal.sosfilt(sos, qij[:,1,1])
    qij[:,1,2] = signal.sosfilt(sos, qij[:,1,2])
    qij[:,2,2] = signal.sosfilt(sos, qij[:,2,2])
    
    return qij


#============ Name of the model
name="R4E1FC_L"
D = D10kpc # Default distance, 10kpc in m
fs=16384
hf = h5py.File(name+'.h5', 'r')
hf.keys()
a_group_key = list(hf.keys())[0]
A = np.array(hf[a_group_key])


# Time: 
if   name == 'mesa20':
    to = A[:,0]
elif name == 'R1E1CA_L' or name == 'R3E1AC_L' or name == 'R4E1FC_L':
    to = A[:,0]
    to = to - to[0]
elif name == 'L15-3' or name == 'N20-2' or name == 'W15-4':
    to = A[:,1]
    to = to - to[0]
# ================== Create quadrupole moment array for the before processing
# The quadrupole is given at the source in cm, so it needs to be brought to the distance of 10kpc
qijo = np.zeros((len(to),3,3),dtype=float)

if   name == 'mesa20':
    qijo[:,0,0] = A[:,1] * cm2kpc / 10.0
    qijo[:,0,1] = A[:,2] * cm2kpc / 10.0
    qijo[:,1,1] = A[:,3] * cm2kpc / 10.0
    qijo[:,2,0] = A[:,4] * cm2kpc / 10.0
    qijo[:,2,1] = A[:,5] * cm2kpc / 10.0
    qijo[:,2,2] = A[:,6] * cm2kpc / 10.0
    qijo[:,1,2] = qijo[:,2,1]
    qijo[:,0,2] = qijo[:,2,0]
    qijo[:,1,0] = qijo[:,0,1]
elif name == 'R1E1CA_L' or name == 'R3E1AC_L' or name == 'R4E1FC_L':
    qijo[:,0,0] = A[:, 5] * cm2kpc / 10.0
    qijo[:,0,1] = A[:, 6] * cm2kpc / 10.0
    qijo[:,1,1] = A[:, 8] * cm2kpc / 10.0
    qijo[:,2,0] = A[:, 7] * cm2kpc / 10.0
    qijo[:,2,1] = A[:, 9] * cm2kpc / 10.0
    qijo[:,2,2] = A[:,10] * cm2kpc / 10.0
    qijo[:,1,2] = qijo[:,2,1]
    qijo[:,0,2] = qijo[:,2,0]
    qijo[:,1,0] = qijo[:,0,1]
    
    
    t, qij = sn_resample_quad(to, qijo, fs)
    
    # Impose trace(qijo) = 0
    trace = qij[:,0,0] + qij[:,1,1] + qij[:,2,2]
    qij[:,0,0] = qij[:,0,0] - 1.0/3*trace
    qij[:,1,1] = qij[:,1,1] - 1.0/3*trace
    qij[:,2,2] = qij[:,2,2] - 1.0/3*trace
        
    # Apply a high-pass Butterworth digital filter
    qij = sn_butfilt_quad(qij,5,fs)
    
    nleft  = 0.5
    nright = 0.0
    t   = sn_remove_edges_wave(t,   nleft, nright)
    qij = sn_remove_edges_quad(qij, nleft, nright)


    
# Create waveform at the equator with original data
(phi_eq, theta_eq) = (0.0, np.pi/2.0) # Equator
hpo, hco = sn_create_waveform(qijo,phi_eq,theta_eq)

# =============================================================================
# Spectrogram
# =============================================================================
plt.figure()
plt.pcolormesh(txz, fxz, Sxz,shading="auto")
plt.plot(tz,fz,"r")
#plt.title("Spectrogram of the signal OG" + str(OG),fontsize=18)
cbar = plt.colorbar()
#cbar.set_label('PSD')
plt.ylim([0,max(fz) +200])
plt.ylabel('Frequency [Hz]',fontsize=15)
plt.xlabel('Time [s]',fontsize=15)
plt.xlim(0,0.7)
plt.tight_layout()
plt.savefig("Spec_OG_"+ str(OG) + ".png",dpi=400)
plt.show()
# =============================================================================
# Other generations of waveforms
# =============================================================================

def AnWF(hrss,alpha,tau,f_0,phi_0):
    """
    hrss = 
    alpha = 
    tau = Width of the signal
    f_0 = Central Frequency
    phi_0 = Central phase random
    """
    # hrss = 1e-25#1e-23
    # alpha = (np.pi/4)
    # tau = .01                           #Width of the signal
    # f_0 = 100                           #Central Frequency
    t_0 = t[int(len(t)/2)]              #Central Time
    # phi_0 = 0 #np.pi/2                  #Central Phase random
    Q = np.sqrt(2)*np.pi*tau
    sqrt_ppn = 4*f_0*np.sqrt(np.pi)
    sqrt_ppd = Q*(1 + np.cos(2*phi_0)*np.exp(-Q**2))
    sqrt_ppdx = Q*(1 - np.cos(2*phi_0)*np.exp(-Q**2))
    sqrt_part = np.sqrt(sqrt_ppn/sqrt_ppd)
    sqrt_partx = np.sqrt(sqrt_ppn/sqrt_ppdx)
    exp_part = np.exp((-(t-t_0)**2)/(tau**2))
    hp_SG = np.cos(alpha)*hrss*sqrt_part*np.cos(2*np.pi*f_0*(t-t_0) + phi_0)*exp_part
    hx_SG = np.sin(alpha)*hrss*sqrt_partx*np.sin(2*np.pi*f_0*(t-t_0) + phi_0)*exp_part
    hp_GS = np.cos(alpha)*(hrss/(np.sqrt(tau)))*exp_part*(2/(np.pi))**(1/4)
    hx_GS = np.sin(alpha)*(hrss/(np.sqrt(tau)))*exp_part*(2/(np.pi))**(1/4)  
    
    return hp_SG,hx_SG,hp_GS,hx_GS

hp_SG,hx_SG,hp_GS,hx_GS = AnWF(1e-25,(np.pi/4),0.01,100,0)

t = t + 0.027  #Alineado Schiedegger con Analítica
to = to + 0.08              # Alineación con Fenomenológico
t = t + 0.08                # Alineación con Fenomenológico



# def padSCH_SG_zeros():
        
#                             # padding zeros to h (SG and Scheid.)
#                             # padding zeros to time (SG and Scheid.)



# ======================== Plot the waveforms
# plt.figure()
# #plt.plot(to,hpo,label='hplus')
# #plt.plot(t1,h1,"black",label="h_Fen")
# plt.plot(to,hco,label='hcross')
# #plt.plot(t,hx_SG,label="hcrossSG")
# plt.title("{0:s}".format(name))
# plt.xlabel('Time [s]')
# plt.ylabel('Strain')
# plt.xlim(0,.5)
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("Schiedegger.png",dpi=400)
# plt.show()

# plt.figure()
# #plt.plot(to,hpo,label='hplus')
# plt.plot(t1,h1,"black",label="h_Fen")
# plt.plot(to,hco,label='hcross')
# plt.plot(t,hx_SG,label="hcrossSG")
# plt.title("Comparativa Modelos", fontsize=18)
# plt.xlabel('Time [s]')
# plt.ylabel('Strain')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("Comparative.png",dpi=400)
# plt.show()

# plt.title("SineGaussian Waveform", fontsize=18)
# #plt.plot(t,hp_SG,label="hplus")
# plt.plot(t,hx_SG,label="hcross",color="darkorange")
# #plt.plot(t,exp_part,label="Exp")
# plt.xlabel('Time [s]')
# plt.ylabel('Strain')
# plt.grid()
# plt.legend()
# plt.savefig("SineG.png",dpi=400)
# plt.show()

# plt.title("Gaussian Signals")
# plt.plot(t,hp_GS,label="hplus")
# #plt.plot(t,hx_GS,label="hcross")
# plt.xlabel('Time [s]')
# plt.ylabel('Strain')
# plt.grid()
# plt.legend()
# plt.show()

datogAWF = pd.DataFrame()
datogAWF["TimeSG"] = t
datogAWF["SignalSG"] = hx_SG
datogAWF.to_csv("DatogAWF"+str(OG)+".csv",header=True,index = False)
dataschie = pd.DataFrame()
dataschie["Time"] = to
dataschie["Signal"] = hco
dataschie.to_csv("Dataschie"+str(OG)+".csv",header=True,index=False)
# =============================================================================
# 
# =============================================================================

datafen = pd.read_csv("Datog3.csv",header=0)
txz = datafen["Time"]
fxz = datafen["Frequency"]

data = pd.read_csv("L1_02.txt",header = None,delim_whitespace=True)
data1 = pd.read_csv("H1_02.txt",header = None,delim_whitespace=True)
data2 = pd.read_csv("L1_03.txt",header = None,delim_whitespace=True)

L1_02t = data[0]
H1_02t = data1[0]
L1_03t = data2[0]

L1_02f = data[1]
H1_02f = data1[1]
L1_03f = data2[1]


# =============================================================================
# Calculate Characteristic Strain.
# =============================================================================
# Characteristic strain for waveforms
fchar, hchar_wave_hc = sn_hchar_wave(t,hx_SG)
fenchar, fhchar_wave = sn_hchar_wave(t1,h1)
# Smoothen the characteristic strains
hchar_wave_hc_med = sn_medfilt_wave(hchar_wave_hc,5)
fehchar_wave_h_med = sn_medfilt_wave(fhchar_wave,5)

# To be able to plot characteristic strain together with detector noise, 
# the characteristic strain need to be multiplied by -1/2 power (check units)

spect_wave_hc = hchar_wave_hc_med * fchar**(-0.5)
spect_wave_hf = fehchar_wave_h_med * fenchar**(-0.5)



#plt.title("Strain Sensitivity", fontsize=18)
plt.loglog(H1_02t,H1_02f,color="blue",label="O2_H1")
plt.loglog(L1_02t,L1_02f,color="Orange",label="O2_L1")
#plt.loglog(L1_03t,L1_03f,color="green",label="O3_L1")
#plt.plot(fchar, spect_wave_hc,label='hcross SineGaussian',color="red")
plt.plot(fenchar, spect_wave_hf,label='h Fenom',color="purple")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Strain")
plt.xlim(10,4.0e3)
plt.ylim([1.0e-24,7.0e-21])
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("OG"+str(OG)+"sensit.png",dpi=400)
# =============================================================================
#  Plots
# =============================================================================

# =============== Spectrogram Graph
# plt.figure()
# plt.pcolormesh(txz, fxz, Sxz,shading="auto")
# plt.plot(tz,fz,"r")
# plt.title("Spectrogram of the signal OG" + str(OG))
# cbar = plt.colorbar()
# #cbar.set_label('PSD')
# plt.ylim([0,max(fz) +200])
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.tight_layout()
# plt.savefig("Spec_OG_"+ str(OG) + ".png",dpi=400)
# plt.show()


plt.plot(tz,fz,color="darkblue")
#plt.title("Frequency (OG" + str(OG) + ")")
plt.ylabel("Frequency [Hz]",fontsize=15)
plt.xlabel("Time [s]",fontsize=15)
plt.xlim(0,.8)
#plt.ylim(-1,1)
plt.grid()
plt.savefig("frecuency_" + str(OG) + ".png",dpi=400)
plt.show()


plt.plot(t_n,a_n1,"og")
#plt.title("Uniform Distribution of a_n (OG" + str(OG) + ")")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
#plt.xlim(-.5,1.5)
plt.ylim(-1,1)
plt.grid()
plt.savefig("a_n.png",dpi=400)
plt.show()

#========== Sample Graph
plt.plot(t1,h1,"black")
plt.title("String waveform of the signal OG" + str(OG))  # Sample at 16386Hz
plt.xlim(-0.5,1.0)
plt.xlabel("t [s]")
plt.ylabel("h/$h_{rms}$")
plt.grid()
plt.savefig("Strain_"+ str(OG) +"_4.png",pad_inches=20,dpi=400)
plt.show()



# ============ Test
# fig = plt.figure(constrained_layout=True)
# gs = gridspec.GridSpec(1,2,figure=fig)
# fig.suptitle("OG" + str(OG))
# ax1 = fig.add_subplot(gs[0,0])
# ax1.plot(t1,h1,color="darkblue")
# ax1.set_xlabel("t [s]")
# ax1.set_ylabel("$h/h_{rms}$")
# ax2 = fig.add_subplot(gs[0,1])
# ax2.set_ylim(0,max(fz) + 200)
# ax2.set_xlim(-.5,1.5)
# ax2.set_xlabel("t [s]")
# ax2.set_ylabel("Frequency [Hz]")
# ax2.plot(tz,fz,"w")
# im1=ax2.pcolormesh(txz, fxz, Sxz,shading="auto")
# plt.colorbar(im1, ax=ax2)
# fig.savefig("OG"+str(OG)+"test.png",dpi=400)





# =============================================================================
# CSV exportation
# =============================================================================

# ========================== Data frame to export
datog1 = pd.DataFrame()

save(datog1) # Call the function to make the csv file
