# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import queue as Queue
import time
import sys

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import ones
from scipy import signal
from scipy import integrate
import threading

from numpy import mean
from numpy import power
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
import bitarray
from  scipy.io.wavfile import read as wavread
import newax25 as ax25

import multiprocessing

from math import gcd
import sounddevice as sd
import RPi.GPIO as GPIO
from functools import reduce
from numpy import ones,zeros, pi, cos, exp, sign


import socket
import os
import sys


# function to compute least common multipler
def lcm(numbers):
    return reduce(lambda x, y: (x*y)//gcd(x,y), numbers, 1)


import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_float

array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library("./libpll", ".")
libcd.pll.restype = c_int
libcd.pll.argtypes= [array_1d_int, c_int, array_1d_int,array_1d_int,  array_1d_int,array_1d_int, c_int, c_float]



  
# Copy the TNC class between these lines:

#     ----------------------------------------------------






















#-----------------------------------------------------------











GPIO.setmode(GPIO.BOARD)
PTT = 12
GPIO.setup(PTT, GPIO.OUT, initial = 0)

print(sd.query_devices())


builtin_idx = 0
usb_idx = 2
sd.default.samplerate=48000
sd.default.channels = 1

print("Using devices: Builtin",builtin_idx, "  USB:",usb_idx)


# this is a function that runs as a thread. It take packet information from
# an APRS client through a TCP socket, creates a valid APRS packet from it 
# modulates it and plays the audio while keying the radio. 

def xmitter():
    modem = TNCaprs(fs = 48000)
    prefix = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(40,)).tolist())
    suffix = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(40,)).tolist())
    while(1):
        data = connection.recv(512)
        if data[:2] == b'\xc0\x00' :
                bits = modem.KISS2bits(data[2:-1])
                sig = modem.modulate(modem.NRZ2NRZI(prefix + bits + suffix))
                GPIO.output(PTT, GPIO.HIGH)
                time.sleep(0.4)
                sd.play(sig*0.15,samplerate=48000,device=usb_idx,  blocking=True)
                GPIO.output(PTT, GPIO.LOW)
        
    print(data)




# Create thread for transmitter
txer = threading.Thread(target = xmitter)# Callback for receiving audio from radio and sotring the samples in a Queue

def queuereplay_callback(indata,outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata
    Qin.put_nowait( indata.copy()[:,0] )  # Global queue

# code to get IP address
gw = os.popen("ip -4 route show default").read().split()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((gw[2], 0))
ipaddr = s.getsockname()[0]
gateway = gw[2]
host = socket.gethostname()
print ("IP:", ipaddr, " GW:", gateway, " Host:", host)
s.close()



# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the port
server_address = (ipaddr, 8001)
print (sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)
sock.listen(1)


connection, client_address = sock.accept()


# Start transmitter

txer.start()



#Create stream and Queue

Qin = Queue.Queue()

st = sd.Stream( device=(usb_idx, builtin_idx),callback=queuereplay_callback)




# Start listening audio. 

st.start()




# Start receiver


Abuffer = 512
Nchunks = 1
fs =48000
#modem = OLDTNCaprs(fs = fs,Abuffer = Abuffer,Nchunks = Nchunks)
#modem = TNCaprs(fs = fs,Abuffer = Abuffer,Nchunks = Nchunks, dec = 4)


modem = TNCaprs(fs = fs,Abuffer = Abuffer,Nchunks = Nchunks)
npack = 0
while (1):
        counter = 0
        while (Qin.empty()):
                if counter == 10:
                    st.stop()
                    st.close()
                    st = sd.Stream( device=(usb_idx, builtin_idx),callback=queuereplay_callback)
                    st.start()
                    counter = 0
                counter = counter + 1
                time.sleep(0.1)
                
        packets  = modem.processBuffer( Qin.get())
        for pkt in packets: 
            
            npack = npack + 1
            try:
                ax = modem.decodeAX25(pkt)
                #print(npack)
                infostr = "%(n) 2i) | DEST: %(dest)s | SRC: %(src)s | DIGI: %(digi)s | %(info)s |" % {
                        'n': npack,
                        'dest': ax.destination,
                        'src': ax.source,
                        'digi': ax.digipeaters,
                        'info': ax.info.decode("ascii").strip()
                    }
                print(infostr)
            except:
                print(npack,"packet")
            msg = b'\xc0\x00'+ modem.bits2KISS(pkt) +  b'\xc0'
            connection.sendall(msg)
            

