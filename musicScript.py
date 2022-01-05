import mido
import time
import os

#v nov format spremeniti
#zanima nas le note_on, time, note

#time atribut je podan v TICKS
def readSong(text):
    mid = mido.MidiFile("music/"+text, clip=True)
    #port = mido.open_output()
    #print(mido.ports)
    return mid
    #velik cajta slo stran ker sem glupo sestavljal midi play omfg, mid.play() sam pretvarja cas med igranjem in ima prakticno timeout
    #print("Playing msg: ",pretvori_v_noto(msg))
    #extractSongInfo(mid)
    #printSongInfo(mid)



def extractSongInfo(mid):

    #tempo - Vecji tempo, pocasneje
    #120BPM je 500ms
    #60BPM je 1s
    #tempo = 60/BPM v sekundah
    #BPM = 60/tempo

    #ticks_per_quarter = <PPQ from the header>  (960)  mid.ticks_per_beat
    #µs_per_quarter = <Tempo in latest Set Tempo event>  (909089)
    #µs_per_tick = µs_per_quarter / ticks_per_quarter  (909089/960 = 63,36)
    #seconds_per_tick = µs_per_tick / 1.000.000  (63,36 / 1.000.000)
    #seconds = ticks * seconds_per_tick (99840 * (63,36 / 1.000.000) = 6.3s)
    #ticks = seconds/seconds_per_tick


    #tick je time parameter v MIDI formatu
    tr = []
    meta_msgs = []

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if not msg.is_meta:
                tr.append((msg.dict().get("type"),msg.dict().get("note"),msg.dict().get("time")))
            if msg.is_meta:
                meta_msgs.append(msg.dict())
                #print(msg.dict())
                #tr.append()
                #print(msg.st)
    #print(tempo_changes)
    printSongInfo(mid)
    timeSum = 0
    for x in tr:
        if x[1] is not None:
            timeSum+=(x[1])
    #print(timeSum)
    #calculateLength(mid, tempo_changes)

    tempo_set = []
    for x in meta_msgs:
        if x["type"]=="set_tempo":
            print("tempo set: ",mido.tick2second(1, mid.ticks_per_beat, x["tempo"])," pri casu ",x["time"])
            tempo_set.append((x["tempo"],x["time"]))

    print(tempo_set)
    #print(tr)
    print(mid.ticks_per_beat)
    print(mido.tick2second(1, mid.ticks_per_beat, 500_000))

    extract_notes(tr,tempo_set,mid.ticks_per_beat)
    #printTickLength(mid)

#ta funkcija gre po vrsti po delta casu znotraj midi datoteke in "prebira" note v casovnem zaporedju, naceloma se tukaj dela izris blokcev
def extract_notes(track,t2s,ticks_per_beat):

    #t2s je tempo change array
    #track je array vseh not znotrja glasbe (non-meta message) (ime_akcije, tipka, cas)

    current_play = [0]*88

    zacet = time.time();
    tempo_timer, tempo_timer_indeks, celoten_tempo, tempo = 0 , 1 , 0, 500_000 #timer za delta timer, indeks za array tempo set, default tempo

    print("tempo casi, ",t2s)

     #default tempo
    if len(t2s)>0 and t2s[0][1] == 0:
        tempo = t2s[0][0] #trenutni tempo
    print("zacetni tempo ", tempo, " zacetni tempo timer ",tempo_timer)
    print(t2s)
    #ce je cas 0, potem naj appenda vse v list za komande, ki se poslejo naprej v funkcijo za prebiranje komand
    komande = []
    #prve komande loop
    for tr in track:
        if tr[2]>0:
            break
        komande.append(tr)
    preberiKomande(current_play,komande)
    for i in range(len(track)):
        #if track[i][0]=="note_on" or track[i][0]=="note_off":
        if track[i][2]>0:
            #ce je komanda, ki ima delay, naj prebira dokler ni se ena komanda z delayom
            j = i+1
            komande.append(track[i])
            while(j<len(track) and track[j][2]==0):
                komande.append(track[j])
                #print("dodajam ", track[j])
                j+=1


            preberiKomande(current_play,komande)
            print(current_play)
            komande.clear()
            tempo_timer+=track[i][2]
            #print(track[i])
            #print("ayo timer ++ ",tempo_timer)
            if(tempo_timer_indeks<len(t2s)):
                #print(tempo_timer,t2s)
                if(t2s[tempo_timer_indeks][1] <= tempo_timer):
                    #print("TEMPO CHANGE ",t2s[tempo_timer_indeks-1]," -> ",t2s[tempo_timer_indeks], " @",tempo_timer)
                    tempo, celoten_tempo, tempo_timer = t2s[tempo_timer_indeks][0], tempo_timer+celoten_tempo, tempo_timer-t2s[tempo_timer_indeks][1] #tempo set kokr je znotraj arraya  #odsteje trenutni cas in delta cas za tempo
                    tempo_timer_indeks+=1 #gleda naslednji element
                    #print("current tempo ", tempo, " current tempo timer ",tempo_timer, ", indeks",tempo_timer_indeks)
            #print("sleep time: ", track[i][2] * tempo/1_000_000_000)
            time.sleep(track[i][2]*tempo/1_000_000_000)
        #print(track[i])
    #ta je na koncu, saj se lahko nahajajo komande po zadnjem timu (note-off za druge note, ko je le ena napisana na delay)
    #preberiKomande(komande)
    #print(time.time()-zacet)
    #print(celoten_tempo)

def pretvori_v_noto(i):
    note = ["C", "C#", 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if (i[0] == "note_on" or i[0] == "note_off"):
        m = note[int(i[1]) % 12]
        n = m+""+str(int(int(i[1])/12))+" "+i[0]+" "+str(i[2])
        return n

def preberiKomande(current_play,com):

    prvi = com[0][0]
    for i in com:

        if i[0]!=prvi: #Tukaj se bo poslal signal, naj se spusti tipko in ponovno pritisne
            prvi=i[0]
            #send current
            print(prvi)
        if(i[0]=='note_on'):
            current_play[i[1] - 9]=1
        elif(i[0]=='note_off'):
            current_play[i[1]-9]=0
        pretvori_v_noto(i)
    #os.system('cls' if os.name == 'nt' else 'clear')


def printTickLength(mid):
    tick_length=0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if not msg.is_meta:
                tick_length+=msg.dict().get("time")

    print(tick_length)


def calculateLength(mid,tempo_changes):
    prejsni_tempo = 0
    tempo_sum = 0
    for x in tempo_changes:
        if x.get("type") == "set_tempo":
            prejsni_time = x.get("time")
            tempo_sum += (prejsni_tempo / mid.ticks_per_beat) / 1000000 * prejsni_time
            prejsni_tempo = x.get("tempo")

        if x.get("type") == "end_of_track":
            tempo_sum += (mid.length-tempo_sum)#*(prejsni_tempo / mid.ticks_per_beat) / 1000000
    return tempo_sum


def play_song():
    pass

def printSongInfo(mid):
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            #if not msg.is_meta:
            print(msg)

if __name__=="__main__":
    readSong("bseasy.mid")
    mid = mido.MidiFile("music/bseasy.mid")
    '''
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)
    '''