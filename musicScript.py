import mido
import time

#v nov format spremeniti
#zanima nas le note_on, time, note

#time atribut je podan v TICKS
def readSong(text):
    mid = mido.MidiFile("music/"+text, clip=True)
    extractSongInfo(mid)
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
    print(tr)
    print(mido.tick2second(1, mid.ticks_per_beat, 500_000))

    extract_notes(tr,tempo_set)
    printTickLength(mid)

#ta funkcija gre po vrsti po delta casu znotraj midi datoteke in "prebira" note v casovnem zaporedju, naceloma se tukaj dela izris blokcev
def extract_notes(track,t2s):

    #t2s je tempo change array
    #track je array vseh not znotrja glasbe (non-meta message) (ime_akcije, tipka, cas)
    zacet = time.time();
    tempo_timer = 0 #timer za delta timer
    tempo_timer_indeks = 1 #Iterator za tempo change array
    celoten_tempo = 0
    print("tempo casi, ",t2s)

    tempo = 500_000 #default tempo
    if len(t2s)>0 and t2s[0][1] == 0:
        tempo = t2s[0][0] #trenutni tempo
    print("zacetni tempo ", tempo, " zacetni tempo timer ",tempo_timer)
    for i in range(len(track)):
        #if track[i][0]=="note_on" or track[i][0]=="note_off":
        if track[i][2]>0:
            tempo_timer+=track[i][2]
            print(track[i])
            #print("ayo timer ++ ",tempo_timer)
            if(tempo_timer_indeks<len(t2s)):
                print(tempo_timer,t2s)
                if(t2s[tempo_timer_indeks][1] <= tempo_timer):
                    print("TEMPO CHANGE ",t2s[tempo_timer_indeks-1]," -> ",t2s[tempo_timer_indeks], " @",tempo_timer)
                    tempo = t2s[tempo_timer_indeks][0] #tempo set kokr je znotraj arraya
                    celoten_tempo=tempo_timer+celoten_tempo
                    tempo_timer=tempo_timer-t2s[tempo_timer_indeks][1] #odsteje trenutni cas in delta cas za tempo
                    tempo_timer_indeks+=1 #gleda naslednji element
                    print("current tempo ", tempo, " current tempo timer ",tempo_timer, ", indeks",tempo_timer_indeks)
            time.sleep(track[i][2]*tempo)
            #print("sleep time: ",track[i][2]*t2s)
        #print(track[i])

    print(time.time()-zacet)
    print(celoten_tempo)


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