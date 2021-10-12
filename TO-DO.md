1. Gumb "connect" naredi nov thread (VideoThread)
2. thread gleda celoten video in vraca QPixmap v signalu
3. Signal mora biti povezan z main appom, ki sprejme QPixmap in spremeni QLabel v Appu

4. Gumb "calibrate" začasno ustavi thread (VideoThread) in vzame zadnjo sliko, ki jo je prejel.
    - Izvede ostalo funkcionalnost, kot je dejanska kalibracija slike

------------

1. Music script odpre file in prebere detajle songa
2. Skripta ustvari picture label, kjer s pomočjo "time" knjižnice uporabljam perf_counter za merjenje dolžine note (pretvorjeno v tick I think)
3. naredi se image, ki na vsake tok časa spremeni sliko (recimo 20x na sekundo, torej na vsake 0.05s oziroma 50ms (perf_counter vrne float cifro v sekundah)