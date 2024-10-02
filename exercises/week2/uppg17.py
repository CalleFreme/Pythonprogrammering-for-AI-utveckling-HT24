## Uppgift 17: Introduktion till Exception handling (undantagshantering)

'''
Vi villa bygga våra program så att kan hantera olika situationer, till exempel när vi får fel typ av input från användaren.
Låt säga att du har ett program som tar in två heltal från användaren och utför en division på dessa två tal,
då måste du se till att ditt program kan hantera när täljaren är 0. Vi vill aldrig ha division med 0.
Vi hanterar dessa undantagsfall, Exceptions med `try`- och `except`-satser i vår kod. I andra programmeringsspråk ser man `catch` istället för `try`.
Det finns olika typer av Exceptions, så som `TypeError`, `ValueError`, `IndexError`, `KeyError`.
Ibland använder vi alla typer av errors som en generell `Exception` istället för specifik typ av Error. Detta ger oss dock
eventuellt mindre detaljerad information om vad som gått fel.

Skriv ett program som demonstrerar grundläggande undantagshantering:

1. Be användaren mata in två tal.
2. Försök (`try`) att dividera det första talet med det andra.
3. Hantera `ZeroDivisionError` om användaren försöker dividera med noll. (`except`)
4. Hantera `ValueError` om användaren matar in något som inte är ett tal. (`except`)
'''

def divide_numbers():
    try:
        num1 = float(input("Ange första talet: "))
        num2 = float(input("Ange andra talet: "))
        result = num1 / num2
        print(f"Resultatet är: {result}")
    except ZeroDivisionError:
        print("Fel: Division med noll är inte tillåtet.")
    except ValueError:
        print("Fel: Vänligen ange giltiga numeriska värden.")
    except Exception as e:
        print(f"Ett oväntat fel inträffade: {e}")

divide_numbers()

# Kommentarer:
# Vi använder en try-except struktur för att hantera olika typer av fel.
# ZeroDivisionError fångar specifikt division med noll.
# ValueError fångar fel vid konvertering av input till float.
# Den generella Exception fångar alla andra oväntade fel.