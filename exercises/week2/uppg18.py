## Uppgift 18: Intro Matplotlib - Visualisering

'''
Skapa ett linjediagram som visar temperaturdata för en vecka.
Som "data" räcker det t.ex. att skapa en lista med sju floats, en för varje dags medeltemperatur. ex: [15.5, 16.0, 14.6, 11.9, 15.3, 16.2, 15.7]
Använd matplotlib för att:

1. Plotta temperaturerna.
2. Lägga till en titel och etiketter för x- och y-axlarna.
3. Anpassa linjefärg och stil.
'''

import matplotlib.pyplot as plt

# Temperaturdata för en vecka
temperatures = [15.5, 16.0, 14.6, 11.9, 15.3, 16.2, 15.7]
days = ['Mån', 'Tis', 'Ons', 'Tor', 'Fre', 'Lör', 'Sön']

plt.figure(figsize=(10, 6))
plt.plot(days, temperatures, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title('Temperatur under en vecka', fontsize=16)
plt.xlabel('Dag', fontsize=12)
plt.ylabel('Temperatur (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(10, 18)  # Anpassa y-axelns gränser

for i, temp in enumerate(temperatures):
    plt.annotate(f'{temp}°C', (days[i], temp), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()

# Kommentarer:
# Vi använder matplotlib för att skapa ett linjediagram.
# plt.figure() sätter storleken på diagrammet.
# plt.plot() ritar själva linjen med anpassade stilalternativ.
# Vi lägger till titlar, etiketter och rutnät för tydlighet.
# plt.annotate() används för att lägga till temperaturvärden ovanför varje punkt.