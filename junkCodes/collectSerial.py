#!/usr/bin/env python3
import serial
import struct
serial_port = '/dev/ttyUSB0'
baud_rate = 9600 #In arduino, Serial.begin(baud_rate)
write_to_file_path = "320deg25.txt"

output_file = open(write_to_file_path, "w+")
ser = serial.Serial(serial_port, baud_rate)
while True:
    line = ser.readline()
#    print(str(line))
    output_file.write(str(line))
