## Uppgift 22: Intro decorators

'''
Detta ger en kort introduktion till dekoratorer och deras användning.
Skapa en enkel decorator `timer` som mäter exekveringstiden för en funktion:

1. Implementera decorator `timer`.
2. Använd `time` modulen för att mäta tiden.
3. Applicera dekoratorn på några funktioner med olika exekveringstider.
4. Forska lite på nätet hur dekoratorer fungerar och deras användningsområden.
'''

import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} tog {end_time - start_time:.4f} sekunder att köra.")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(2)
    print("Slow function klar.")

@timer
def fast_function():
    print("Fast function klar.")

slow_function()
fast_function()

# Kommentarer:
# Vi skapar en decorator som mäter exekveringstiden för en funktion.
# functools.wraps bevarar metadata för den dekorerade funktionen.
# Decoratorn använder *args och **kwargs för att hantera alla typer av funktionsargument.
# time.time() används för att mäta tiden före och efter funktionsanropet.
# Dekoratorn appliceras på funktioner med @timer syntax.