## Uppgift 9: Intro till Arv och polymorfism

'''
Skapa en hierarki av djurklasser:

1. Börja med en basklass `Animal` med attributen `name` och `sound`.
2. Skapa subklasser `Dog`, `Cat`, och `Cow` som ärver från `Animal`.
3. Överskugga `make_sound()` metoden i varje subklass för att returnera djurets specifika ljud.
4. Skapa en funktion `animal_chorus(animals)` som tar en lista av djur och låter alla göra sitt ljud.
'''

class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound
    
    def make_sound(self):
        return f"{self.name} säger {self.sound}!"

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name, "voff")

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "mjau")

class Cow(Animal):
    def __init__(self, name):
        super().__init__(name, "mu")

def animal_chorus(animals):
    for animal in animals:
        print(animal.make_sound())

# Testa Animal classes
animals = [Dog("Fido"), Cat("Whiskers"), Cow("Bessie")] # Lista av djur
animal_chorus(animals)

# Kommentarer:
# Vi använder en basklass Animal och subklasser för specifika djur.
# super().__init__() används för att anropa basklassens konstruktor.
# Polymorfism demonstreras i animal_chorus() funktionen, som kan hantera alla typer av Animal-objekt.