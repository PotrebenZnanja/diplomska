import pygame
import mido
import rtmidi
import time

#perf_counter za measure ticks

def runtime_test(n):
    print("runtime_test funkcije ")
    start_time = time.time_ns();
    L = []
    for i in range(1, n):
        if i % 3 == 0:
            L.append(i)
    print("Done ",L[-1])
    print(time.time_ns()-start_time)
    L.clear()

    start_time = time.time_ns()
    L = [i for i in range (1, n) if i%3 == 0]
    print("Done ",L[-1])
    print(time.time_ns()-start_time)

def declaration_runtest():
    start = time.time_ns();
    a = 2
    b=3
    c=4
    d=0
    h= -5121515
    print(a+b+c+d+h)
    print(time.time_ns()-start)
    start = time.time_ns();
    a,b,c,d,h = 2,3,4,0,-5121515
    print(a + b + c + d + h)
    print(time.time_ns() - start)


if __name__ == "__main__":
    prev_timer = time.time_ns()
    i = 0
    while(i<15):
        timer = time.time_ns()
        print(timer-prev_timer)
        prev_timer = timer
        i+=1


    print("process_time")
    prev_timer = time.process_time_ns()
    i = 0
    while (i < 15):
        timer = time.process_time_ns()
        print(timer - prev_timer)
        prev_timer = timer
        i += 1

    print("perf_counter")
    prev_timer = time.perf_counter_ns()
    i = 0
    while (i < 15):
        timer = time.perf_counter_ns()
        print(timer - prev_timer)
        prev_timer = timer
        i += 1

    runtime_test(10000000)
    print("declaration runtest")
    declaration_runtest()

