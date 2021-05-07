# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:15:58 2021

@author: joonahuh
Joonatan Huhtasalo
joonatan.huhtasalo@helsinki.fi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as scp
from matplotlib import rc
import os

rc('text', usetex=True)

def data_unpack_csv(fileName):
    
    
    datafile = pd.read_csv(fileName, delimiter=';', decimal=',')
        
    return datafile

def data_unpack_txt(fileName, headers):
    
    datafile = pd.read_csv(fileName, delimiter='\t', header=headers)
    
        
    return datafile

def sin(x,a,b,c):
    return a * np.sin( b * x + c )

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def regression(x, L, R):
    V=2.47
    #R=10+38
    w=2*np.pi*x
    C=10**(-5)

    return V/np.sqrt(R**2+(w*L-1/(w*C))**2)
    

master_data = pd.DataFrame(columns=['Taajuus (Hz)','Sähkövirran amplitudi (A)'])

usrInpt = 0

max_pot = 0

data_collection_flag = False

usrInpt = input('Run the data collection (y/n): ')

if( usrInpt == 'y'):
    data_collection_flag = True
    runs = int(input('How many data runs (int): '))
elif( usrInpt == 'n'):
    data_collection_flag = False
else:
    print('Command not recognised!')
    

if( data_collection_flag ):
    for k in range(runs):
        usrInpt = input('Starting frequency, step size and final frequency and folder prefix: ')
        freq = int(usrInpt.split(' ')[0])
        step = int(usrInpt.split(' ')[1])
        final_freq = int(usrInpt.split(' ')[2])
        
        folder_prefix = usrInpt.split(' ')[3]
        
        for i in range(freq, final_freq + step, step):
            #freq = 50
            
            
            data_path = folder_prefix + '/' + str(i) + 'hz.csv'
            
            data = data_unpack_csv(data_path)
            
            fit, fitcov = opt.curve_fit(sin,
                                        data.iloc[:,0],
                                        data.iloc[:,2],
                                        [1,2*np.pi/(1/i),1])
            
            perr = np.sqrt(np.diag(fitcov))
            
            # print(data)
            print(fit)
            
            if max(data.iloc[:,1]) > max_pot:
                max_pot = max(data.iloc[:,1])
            
            df = pd.DataFrame([[i,np.abs(fit[0]),np.nan,perr[0]]],
                              columns=['Taajuus (Hz)','Sähkövirran amplitudi (A)','Sähkövirran amplitudi (mA)','Virhe'])
            master_data = master_data.append(df, ignore_index=True)
            master_data['Sähkövirran amplitudi (mA)'] = master_data['Sähkövirran amplitudi (A)']*10**3
            # master_data['Virhe'] = 
            # fit_linear_spacing = np.linspace(0,0.1,200)
            # df = pd.DataFrame([i,perr],columns=['Virhe'])
            # master_data = master_data.append(df, ignore_index=True)
            
            # fig, ax = plt.subplots(figsize=(10,6))
            
            # ax.plot(data.iloc[:,0],
            #          data.iloc[:,2],
            #          c='b')
            
            # ax.plot(fit_linear_spacing,
            #         sin(fit_linear_spacing,fit[0],fit[1],fit[2],fit[3]),
            #         c='r')

    master_data.sort_values('Taajuus (Hz)', inplace=True, ignore_index=True)
    print(master_data)

print('Maksimi potentiaali: ' + str(max_pot))

if data_collection_flag:
    master_data = master_data.drop([14])
    master_data.to_csv('master_data.csv', index=False)

master_data = pd.read_csv('master_data.csv', delimiter=',')

data = data_unpack_txt("RLCdata.txt", [0,1,2])

print(master_data)



fig, ax = plt.subplots(1,1, figsize=(10,6))

freq_err = data.columns[0][2].split()[1]
amp_err = data.columns[1][2].split()[1]


ax.grid(True)
print(data['taajuus'])


x = np.linspace(50,250,200)

# popt, pcov = opt.curve_fit(gaussian,
#                         master_data.iloc[:,0],
#                         master_data.iloc[:,2],
#                         [80, 165, 20])

popt, pcov = opt.curve_fit(regression,
                           master_data.iloc[:,0],
                           master_data.iloc[:,1],
                           [0.1,30])

perr = np.sqrt(np.diag(pcov))

ax.plot(master_data['Taajuus (Hz)'],
        master_data['Sähkövirran amplitudi (mA)'],
        c='b',
        linestyle='none',
        marker='.',
        label='Data')


# ax.plot(x,
#         gaussian(x,popt[0],popt[1],popt[2]),
#         c='r')

fit = 10**3*regression(x,popt[0],popt[1]) #   Muutetaan fitti 10^3 mA yksiköihin

ax.plot(x,
        fit,
        c='r',
        label='Teoreettinen sovitus')
props = dict(arrowstyle="-",
             connectionstyle="arc3,rad=0")

plt.rcParams.update({'font.size': 14})
annotateText = (r'Fit: $\tilde{I}_{0}=\frac{ \tilde{V}_{0} }{ \sqrt{R^{2}+(2\pi f L-1/2C\pi f)^{2}} }$' + '\n'
                + r'Data error: $ \tilde{I}_{0} \pm' + "{:.3f}".format(max(master_data['Virhe'])*10**3) + '\;\mathrm{mA}$' + '\n'
                + r'$\tilde{I}_{0,max}=' + "{:.2f}".format(max(fit)) + '\pm'
                + "{:.3f}".format(max(master_data['Virhe'])*10**3) + '\;\mathrm{mA}$')

bbox = dict(pad=10, fc="pink")
ax.annotate(annotateText,
            xy=(140,10**3*regression(140,popt[0],popt[1])),
            xytext=(50,50),
            arrowprops=props,
            bbox=bbox)

max_i = "{:.2f}".format(max(fit))
max_f = "{:.2f}".format(x[np.where(fit == max(fit))][0])
print('Max virta: ' + max_i + '+/-' + "{:.3f}".format(max(master_data['Virhe'])*10**3) + 'mA\nTaajuus: ' + max_f)
print('Resistanssi: ' + str("{:.2f}".format(np.abs(popt[1]))) + '+/-' + str("{:.4f}".format(perr[1])) + ' Ohmia')
# print('Jännite: ' + str("{:.2f}".format(np.abs(popt[1]))))
print('Induktanssi: ' + str("{:.2f}".format(np.abs(popt[0]))) + '+/-' + str("{:.4f}".format(perr[0])) + ' H')

ax.axvline(x[np.where(fit == max(fit))][0],
         c='black',
         linestyle='dashed',
         label='Resonanssitaajuus: ' + max_f)

ax.errorbar(master_data['Taajuus (Hz)'],
            master_data['Sähkövirran amplitudi (mA)'],
            xerr=1,
            yerr=max(master_data['Virhe'])*10**3,
            linestyle='none')

ax.legend()
#ax.errorbar(data)
ax.set_xlabel( str(data.columns[0][0]) + ' '
              + str(data.columns[0][1]) ,fontsize=14)
ax.set_ylabel( str(data.columns[1][0]) + ' '
              + str(data.columns[1][1]) ,fontsize=14)
ax.set_title( 'Sähkövirran amplitudi taajuuden funktiona',fontsize=14 )

fig.savefig('plotti.pdf',dpi=150)
































