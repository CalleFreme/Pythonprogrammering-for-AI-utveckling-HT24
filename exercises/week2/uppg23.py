## Uppgift 23: Generators

'''
Implementera en generator som producerar Fibonacci-sekvensen upp till ett givet antal termer.
'''

def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Använd generatorn
for num in fibonacci_generator(10):
    print(num, end=' ')

# Kommentarer:
# Denna generator producerar Fibonacci-tal upp till ett givet antal termer.
# yield används för att returnera varje tal i sekvensen.
# Generators är funktioner specialiserade på att producera en sekvens.
# Vi definierar en en generator genom att använda yield istället för return.
# Generatorn pausar sin exekvering efter varje yield och återupptar där den slutade vid nästa anrop.
# Detta är mer minneseffektivt än att skapa en hel lista, särskilt för stora sekvenser.
