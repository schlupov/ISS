import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import lfilter
from scipy.signal import tf2zpk
from scipy.stats import norm
#---------------------------------------------------------------------------------------------------#
FS, DATA = wavfile.read('./xchlup08.wav')

print("Ukol 1")
print(f"Sampling frequency is {FS} Hz")
print(f"Length: {DATA.size} samples")
print(f"Length: {DATA.size/FS} s")
print(f"Number of represented binary symbols: {DATA.size/16}")
#---------------------------------------------------------------------------------------------------#
def task2(FS, DATA):
    print("Ukol 2")
    shorter_data = DATA[:320]  # for 20 ms
    points = []
    time_for_points = []
    time = []
    for i in range(8, 320, 16):
        time_for_points.append(i / FS)
        if DATA[i] > 0:
            points.append(1)
        else:
            points.append(0)

    for i in range(0, 320):
        time.append(i / FS)
    #plt.plot(time, shorter_data/DATA.size, time_for_points, points,'.')
    #plt.xlabel('t [ms]')
    #plt.ylabel('s [n]')
    #plt.savefig('1.png')

    with open("xchlup08.txt") as f:
        txt_file = f.readlines()

    cmp = []
    for i in range(0, 20):
        to_cmp = txt_file[i].rstrip()
        if int(to_cmp) ^ int(points[i]) == 0:
            cmp.append("ok")

    print("Compare 0 and 1 from .txt file and from my decoding into binary symbols:", set(cmp), u'\u2713')
#---------------------------------------------------------------------------------------------------#
def task3(FS, DATA):
    print("Ukol 3")
    B = [0.0192, -0.0185, -0.0185, 0.0192]
    A = [1.0000, -2.8870, 2.7997, -0.9113]

    z, p, k = tf2zpk(B, A)
    #plt.figure(figsize=(7,7))
    #plt.gca().add_patch(plt.Circle((0,0), radius=1., color='grey', fc='none'))
    #plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    #plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
    #plt.xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    #plt.ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
    #plt.legend(loc='upper right')
    #plt.savefig('2.png', dpi=125)
    #plt.show()
#---------------------------------------------------------------------------------------------------#
def task4(FS,DATA):
    print("Ukol 4")
    B = [0.0192, -0.0185, -0.0185, 0.0192]
    A = [1.0000, -2.8870, 2.7997, -0.9113]

    normf, freqresponse = signal.freqz(B, A)

    f = FS * normf / (2*np.pi)
    #plt.plot(f, np.abs(freqresponse))
    #plt.xlabel("f [Hz]")
    #plt.ylabel("|H(f)|")
    #plt.savefig('3.png', dpi=125)
#---------------------------------------------------------------------------------------------------#
def task5(FS,DATA):
    print("Ukol 4")
    B = [0.0192, -0.0185, -0.0185, 0.0192]
    A = [1.0000, -2.8870, 2.7997, -0.9113]

    time = []
    for i in range(0, 320):
        time.append(i / FS)
    shorter_data = DATA[:320]
    filtered_signal = signal.lfilter(B, A, DATA)
    #plt.plot(time, shorter_data/DATA.size, time, filtered_signal[:320]/DATA.size)
    return filtered_signal
    #plt.savefig('4.png', dpi=125)
#---------------------------------------------------------------------------------------------------#
def task6(FS,DATA):
    print("Ukol 5 a 6")
    filtered_signal = task5(FS,DATA)
    time_for_shifted = []
    for i in range(0, 320):
        time_for_shifted.append(i / FS - 0.0007)

    points = []
    time_for_points = []
    signal_for_points = filtered_signal[:320] / DATA.size

    for i in range(8, 320, 16):
        time_for_points.append(i / FS - 0.0007)
        if signal_for_points[i] > 0:
            points.append(1)
        else:
            points.append(0)

    time = []
    for i in range(0, 320):
        time.append(i / FS)
    shorter_data = DATA[:320]

    #plt.plot(time, filtered_signal[:320]/DATA.size, label='ss[n]')
    #plt.plot(time, shorter_data/DATA.size, color='dimgrey', label='s[n]')
    #plt.plot(time_for_shifted, filtered_signal[:320]/DATA.size, label='ss$_{shifted}$[n]')
    #plt.plot(time_for_points, points,'.', color='tan', label='decoded symbols')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),  shadow=True, ncol=4)
    #plt.xlabel('t')
    #plt.ylabel('s[n], ss[n], ss$_{shifted}$[n], decoded symbols')
    #plt.savefig('5.png', dpi=125)
#---------------------------------------------------------------------------------------------------#
def task7(FS, DATA):
    print("Ukol 7")
    with open("xchlup08.txt") as f:
        txt_file = f.readlines()

    points = []
    time_for_points = []
    filtered_signal = task5(FS, DATA)
    signal_for_points = filtered_signal[:320] / DATA.size
    for i in range(8, 320, 16):
        time_for_points.append(i / FS - 0.0007)
        if signal_for_points[i] > 0:
            points.append(1)
        else:
            points.append(0)

    cmp = []
    for i in range(20):
        to_cmp = txt_file[i].rstrip()
        if int(to_cmp) ^ int(points[i]) == 0:
            cmp.append("ok")
        else:
            cmp.append("bad")

    count = 0
    for item in cmp:
        if item == "bad":
            count += 1

    print("Chybovost %d%%" % (count/len(cmp)*100))
    print(count)
#---------------------------------------------------------------------------------------------------#
def task8(FS,DATA):
    print("Ukol 8")
    filtered_signal = task5(FS, DATA)
    dft = fft(DATA)
    moduls = np.absolute(dft)
    #plt.plot(moduls[:(FS//2)], color='dimgrey')
    #plt.xlabel('f [Hz]')
    #plt.ylabel(' ')
    #plt.savefig('6.png', dpi=125)

    dft = fft(filtered_signal)
    moduls = np.absolute(dft)
    #plt.plot(moduls[:(FS//2)], color='dimgrey')
    #plt.xlabel('f [Hz]')
    #plt.ylabel(' ')
    #plt.savefig('7.png', dpi=125)
#---------------------------------------------------------------------------------------------------#
def task9(FS, DATA):
    print("Ukol 9")
    N = 1
    xmin = np.min(DATA)
    xmax = np.max(DATA)
    n_aprx = 50
    x = np.linspace(xmin, xmax, n_aprx)
    binsize = np.abs(x[1] - x[0])
    hist, _ = np.histogram(DATA, x)
    px = hist / N / binsize
    #plt.plot(x[:49]/DATA.size, px)
    #plt.gca().set_xlabel('$x$')
    #plt.gca().grid(alpha=0.5, linestyle='--')
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('8.png', dpi=125)
    print(np.trapz(x=x[:49]/DATA.size,y=px))
#---------------------------------------------------------------------------------------------------#
def task10(FS, DATA):
    print("Ukol 10")
    r = np.array([])
    result = []
    for i in range(-50,51):
        r = np.append(r, np.sum((DATA/DATA.size)*shift(DATA, i, cval=0)))
        result.append(i)
    #plt.plot(result, r/10e8)
    #plt.ylabel('R[k]')
    #plt.xlabel('k')
    #plt.show()
    # plt.savefig('9.png', dpi=125)
    return r/10e8
#---------------------------------------------------------------------------------------------------#
def task11(FS, DATA):
    print("Ukol 11")
    Rv = task10(FS, DATA)
    print('Value of coefficient R[0] = {}'.format(Rv[50]))
    print('Value of coefficient R[1] = {}'.format(Rv[51]))
    print('Value of coefficient R[16] = {}'.format(Rv[66]))
#---------------------------------------------------------------------------------------------------#
def task12(FS, DATA):
    print("Ukol 12")
    hist, xedges, yedges = np.histogram2d(DATA, shift(DATA, 1, cval=0), 32, normed=True, range=[[-FS, FS], [-FS, FS]])
    #plt.imshow(hist, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #plt.colorbar()
    #plt.savefig('10.png', dpi=125)
#---------------------------------------------------------------------------------------------------#
def task13(FS, DATA):
    print("Ukol 13")
    hist, xedges, yedges = np.histogram2d(DATA, shift(DATA, 1, cval=0), 32, normed=True, range=[[-FS, FS], [-FS, FS]])
    squareSize = (xedges[1]-xedges[0])*(yedges[1]-yedges[0])
    integral = squareSize * np.sum(hist)
    print(f"Total volume of 2D histogram: {integral}")
#---------------------------------------------------------------------------------------------------#
def task14(FS, DATA):
    print("Ukol 14")
    hist, xedges, yedges = np.histogram2d(DATA, shift(DATA, 1, cval=0), 32, normed=True, range=[[-FS, FS], [-FS, FS]])
    squareSize = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
    x = np.linspace(-16000, 16000, num=32)
    x = np.tile(x, (32, 1))
    r = np.sum(x * x.transpose() * hist) * squareSize
    print(f"Histogram calculated R[1] = {r/10e8}")


if __name__== "__main__":
  functions = [task2,task3,task4,task5,task6,task7,task8,task9,task10,task11,task12,task13,task14]
  for i in range(13):
      functions[i](FS,DATA)
