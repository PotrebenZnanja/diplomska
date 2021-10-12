import pygame
import mido
import rtmidi
import time

#perf_counter za measure ticks

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