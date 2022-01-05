import pygame
import mido

pygame.init()
BLACK = [  0,   0,   0]
WHITE = [255, 255, 255]
note_list = []
note_list_off = []

SIZE = [380, 380]
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Python MIDI Program by Wilson Chao")
clock = pygame.time.Clock()
done = False
while done == False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True
    for msg in mido.MidiFile("music/bseasy.mid", clip=True).play():
        n= msg.dict().get('note')
        if (n is not None):
            type = msg.dict().get('type')
            x=(n-47)*10 
        #if (msg.velocity>0):
            if type=="note_on":
                note_list.append([x, 0])
            else:
                note_list_off.append([x, 0])  
        #else:       
        #   note_list_off.append([x, 0])   
            for i in range(len(note_list)):
                pygame.draw.circle(screen, WHITE, note_list[i], 10)
                note_list[i][1] += 1  

            pygame.display.flip()    
            for i in range(len(note_list_off)):
                pygame.draw.circle(screen, BLACK, note_list_off[i], 10)
                note_list_off[i][1] += 1  
        #clock.tick(2000) 
    
    pygame.quit ()