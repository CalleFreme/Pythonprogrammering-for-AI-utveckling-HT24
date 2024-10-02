import math

class Math:
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def subtract(a, b):
        return a - b
    
    @staticmethod
    def divide(a, b):
        if b != 0:
            return a / b
        raise ValueError("Cannot divide by zero")
    
    @staticmethod
    def multiply(a, b):
        return a * b
    
    @staticmethod
    def gcd(a, b):
        return math.gcd(a, b)
    
    @staticmethod
    def area_circle(r):
        return math.pi * r ** 2
    
    @staticmethod
    def circumference(d):
        return math.pi * d

# Testa Math class
print(Math.add(5, 3))
print(Math.area_circle(2))

# Kommentarer:
# Här använder vi statiska metoder (@staticmethod) eftersom dessa matematiska operationer
# inte behöver tillgång till någon instansdata, d.v.s information om nåt särskilt objekt.
# Vi använder alltså bara själva klass-definition av Math i detta program; vi skapar inga Math-objekt.
# Vi importerar math-modulen för att använda dess inbyggda matematiska funktioner för pi och gcd.
# Notera att vi lägger till felhantering för division med noll.