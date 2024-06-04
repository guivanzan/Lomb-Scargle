from astropy.timeseries import LombScargle
import numpy as np
from scipy import signal

def freqToNineNinePercent(Depth, Proxy, freq, amp):
    # Parametros para verificar onde se encontra a freq. com 99% da energia do periodograma
    Dt = np.diff(Depth)
    handMean = np.mean(Dt)

    w, po = signal.periodogram(Proxy)
    fd1 = w / handMean

    poc = np.cumsum(po)
    pocnorm = 100 * poc / max(poc)
    poc1 = np.argwhere(pocnorm >= 99)

    if (fd1[poc1[0][0]]/fd1[-1]) <= 0.85:
        ValidNyqFreqR = fd1[poc1[0][0]]
    else:
        ValidNyqFreqR = fd1[-1]
    
    print(f'Freq. to 99% = {ValidNyqFreqR}')

    freq_To_fmax = []
    amp_To_fmax = []

    for j in range(len(freq)):
        if freq[j] > ValidNyqFreqR: break
        freq_To_fmax.append(freq[j])
        amp_To_fmax.append(amp[j])
    
    return freq_To_fmax, amp_To_fmax, ValidNyqFreqR

def LombScargleAstroManual(xdata, ydata, useNyquist = False):
    '''Função que aplica o método LombScargle nos dados. Recomendados para dados
       que possuem intervalos espaçados de forma irregular.
       Retorna frequências e amplitudes.
    '''
    # w são as frequências sob as quais o método calculará as amplitudes.

    LS = LombScargle(xdata, ydata, fit_mean=False, center_data=False, normalization='psd')
    gimmickFreqs, gimmickAmps = LS.autopower(method='fast')
    
    if useNyquist:
        N = len(xdata)
        DT = np.mean(np.diff(xdata))
        Nyquist = 1/(2*DT)

        Fmin = 1/(4*N*DT)
        Fmax = round(Nyquist,4)
        
        freqs = list(np.arange(Fmin, Fmax, Fmin))
        amps = LS.power(frequency=freqs, method='fast', normalization='psd',assume_regular_frequency=True)

        return amps, freqs
    
    else:
        aeDepth = xdata.tolist()
        aeProxy = ydata.tolist()
        freq_99, amp_99, Nyq_99 = freqToNineNinePercent(aeDepth, aeProxy, gimmickFreqs, gimmickAmps)

        N = len(xdata)
        DT = np.mean(np.diff(xdata))

        Fmin = 1/(4*N*DT)
        Fmax = round(freq_99[-1],4)
        
        freqs = list(np.arange(Fmin, Fmax, Fmin))
        amps = LS.power(frequency=freqs, method='fast', normalization='psd')

        return amps, freqs