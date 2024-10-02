## Uppgift 10: Arv och polyformism: Klass och subklasser för geometriska former (med kod-skelett/scaffolding)

'''
Skriv färdigt programmet/klassen.
Komplettera klassen med metoder och funktionalitet enligt kommentarerna:
'''

import math

class GeometricShape:
    def __init__(self, name):
        self.name = name

    def area(self):
        # Grund-definition av metoden är tom, och varnar om en subklass använder
        # metoden utan att först ha definierat den själv.
        raise NotImplementedError("Subclass must implement abstract method")

    def perimeter(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def __str__(self):
        return f"{self.name} med area {self.area():.2f} och omkrets {self.perimeter():.2f}"

class Rectangle(GeometricShape):
    def __init__(self, width, height):
        super().__init__("Rektangel")
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(GeometricShape):
    def __init__(self, radius):
        super().__init__("Cirkel")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

# Testa GeometricShapes
shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print(shape)

# Kommentarer:
# GeometricShape fungerar som en abstrakt basklass med abstrakta metoder.
# GeometricShape definierar vilka metoder som subklasser måste implementera/definiera.
# En abstract basklass innebär att alla subklasser ska ha egna definitioner för alla metoder i basklassen.
# Subklasserna implementerar de abstrakta metoderna area() och perimeter().
# Circle och Rectangle har båda en area och en omkrets (perimeter), men hur de beräknas är olika. Därför behöver de egna definitioner
# för area() och perimeter().
# __str__() metoden i basklassen ger en enhetlig strängrepresentation för alla former.