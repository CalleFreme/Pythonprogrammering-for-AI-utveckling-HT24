'''
Uppgift 14
Skapa ett enkelt gissningsspel där datorn 
väljer ett slumpmässigt tal mellan 1-100 (eller annat intervall), 
och låt användaren gissa tills de hittar rätt nummer.
För varje felaktig gissning berättar datorn om det rätta svaret
är högre eller lägre än spelarens gissning.
Vid rätt gissning får man välja om man vill spela igen.
'''

import random


keep_playing = True

while keep_playing: # Loopa spelet så länge keep_playing är True.
    computer_number = random.randint(1,100) # Börjar varje spel med att ta fram ett random tal
    has_correct_guess = False # I början av spelet har vi inte gjort nån gissning ännu

    while not has_correct_guess:    # Loopa tills has_correct_guess blir True
        
        user_guess = int(input("Guess a number between 1-100: "))
        
        if user_guess > 100 or user_guess < 1:
            
            print("Your guess must be within 1 to 100")
            
            continue # Gå tillbaka till toppen av loopen och låt användaren skriva ny gissning

        if user_guess == computer_number:
            
            print(f"{user_guess} is the correct guess!")
            
            has_correct_guess = True    # Bryter oss ur loopen då villkoret inte längre stämmer
            
            play_again = input("Do you want to play again? Enter Yes/No: ").lower() # Gör om svaret till små bokstäver direkt, för lättare jämförelse i nästa rad
            
            if play_again != "yes":
                keep_playing = False
            else:
                keep_playing = True
            break
        elif user_guess < computer_number:
            print(f"The correct number is higher than your guess...")
        else:
            print(f"The correct number is lower than your guess...")
