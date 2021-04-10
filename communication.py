import socket
import threading
import numpy as np
import time
from pynput import keyboard

global stop_threads
stop_threads = False

class LaserReceiver(object):
    def __init__(self, laser_addr, laser_port):
        self.laser_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.laser_addr = laser_addr
        self.laser_port = laser_port
        self.laser_sock.settimeout(1.0)
        self.laser_sock.bind((self.laser_addr, self.laser_port))
        self.laser_thread = threading.Thread(target=self.receive_laser)
        self.laser_thread.start()
        self.laser_data = np.array([])
        self.timeout = False

    def receive_laser(self):
        while True:
            try:
                data, server = self.laser_sock.recvfrom(65535)
                # self.laser_data = np.frombuffer(data).reshape(-1, 2)
                self.laser_data = np.frombuffer(data)
                self.timeout = False
                # print('Received Laser Data', self.laser_data.shape)
            except socket.timeout:
                self.timeout = True
            # stop thread
            global stop_threads
            if stop_threads:
                break

class PosReceiver(object):
    def __init__(self, pos_addr, pos_port):
        self.pos_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pos_addr = pos_addr
        self.pos_port = pos_port
        self.pos_sock.settimeout(1.0)
        self.pos_sock.bind((self.pos_addr, self.pos_port))
        self.pos_thread = threading.Thread(target=self.receive_pos)
        self.pos_thread.start()
        self.pos_data = np.array([])
        self.timeout = False

    def receive_pos(self):
        while True:
            try:
                data, server = self.pos_sock.recvfrom(4096)
                self.pos_data = np.frombuffer(data).reshape(7,)
                self.timeout = False
                # print('Received Pos Data', self.pos_data)
            except socket.timeout:
                self.timeout = True
            # stop thread
            global stop_threads
            if stop_threads:
                break

class VelReceiver(object):
    def __init__(self, vel_addr, vel_port):
        self.vel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vel_addr = vel_addr
        self.vel_port = vel_port
        self.vel_sock.settimeout(1.0)
        self.vel_sock.bind((self.vel_addr, self.vel_port))
        self.vel_thread = threading.Thread(target=self.receive_vel)
        self.vel_thread.start()
        self.vel_data = np.array([])
        self.timeout = False

    def receive_vel(self):
        while True:
            try:
                data, server = self.vel_sock.recvfrom(4096)
                self.vel_data = np.frombuffer(data)
                self.timeout = False
                # print('Received Vel Data', self.vel_data)
            except socket.timeout:
                self.timeout = True
            # stop thread
            global stop_threads
            if stop_threads:
                break

class CmdSender(object):
    def __init__(self, cmd_addr, cmd_port, detect_keyboard):
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_addr = cmd_addr
        self.cmd_port = cmd_port
        self.detect_keyboard = detect_keyboard
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        if self.detect_keyboard:
            if key == keyboard.Key.up:
                self.send(1.0, 0.0)
            elif key == keyboard.Key.down:
                self.send(-1.0, 0.0)
            elif key == keyboard.Key.left:
                self.send(0.0, 1.0)
            elif key == keyboard.Key.right:
                self.send(0.0, -1.0)

    def send(self, v, w):
        cmd = np.array([float(v), float(w)])
        self.cmd_sock.sendto(cmd.tobytes(), (self.cmd_addr, self.cmd_port))

class Robot(object):
    def __init__(self, host_addr, laser_port, pos_port, robot_addr, cmd_port, detect_keyboard=False):
        self.laser_recv = LaserReceiver(laser_addr=host_addr, laser_port=laser_port)
        self.pos_recv = PosReceiver(pos_addr=host_addr, pos_port=pos_port)
        self.cmd_send = CmdSender(cmd_addr=robot_addr, cmd_port=cmd_port, detect_keyboard=detect_keyboard)

    @property
    def laser(self):
        return self.laser_recv.laser_data

    @property
    def laser_timeout(self):
        return self.laser_recv.timeout   
    
    @property
    def pos(self):
        return self.pos_recv.pos_data

    @property
    def pos_timeout(self):
        return self.pos_recv.timeout
    

    def sendCommand(self, v, w):
        return self.cmd_send.send(v, w)


if __name__ == '__main__':
    stop_threads = False
    # laser1 = LaserReceiver(laser_addr='192.168.10.200', laser_port=50001)
    # pos1 = PosReceiver(pos_addr='192.168.10.200', pos_port=50002)
    host_addr = '192.168.8.8'
    laser_port = [60001]
    pos_port = [60002]
    robot_addr = ['192.168.8.9']
    cmd_port = 60001
    robot1 = Robot(host_addr, laser_port[0], pos_port[0], robot_addr[0], cmd_port, True)
    # robot2 = Robot(host_addr, laser_port[1], pos_port[1], robot_addr[1], cmd_port, True)
    while True:
        try:
            # TODO
            time.sleep(1)
            # print('robot1', robot1.laser.shape, robot1.laser_timeout, robot1.pos.shape, robot1.pos_timeout)
            # print('robot2', robot2.laser.shape, robot2.laser_timeout, robot2.pos.shape, robot2.pos_timeout)
        except KeyboardInterrupt:
            stop_threads = True
            break