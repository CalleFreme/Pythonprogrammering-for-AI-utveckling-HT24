
myString = "banana"

# Vi kan komma åt en "slice" av strängen (eller listan) genom att använda ":"
# Siffran efter ":" är exklusivt, inte inklusivt, det vill säga intervallet 
# går från och MED 0, upp TILL 2 (= upp till och MED 1)
# Om man använder negativa index är det som att räkna index baklänges i strängen (t.ex. -2 blir näst-sista indexet)
frontPart = myString[0:2]   # 0:2 plockar ut tecken från strängens första (index 0) till andra tecken (index 1)
endPart = myString[-2:]     # -2: plockar ut tecken från strängens näst sista index (-2) till slutet (tomt efter ':')
print(frontPart + ' ' + endPart)