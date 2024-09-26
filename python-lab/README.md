# Python-laboration - Deadline 3/10

På denna sida hittar du information och instruktioner till den första betygsgrundande inlämningen, en laboration i Python. Syftet med labben är att ni ska känna er tillräckligt bekväma i Python för att gå vidare till mer avancerade koncept och introduktion av AI i Python.

Uppgiften går ut på att göra ett fungerande, välgjort Python-projekt. Du får välja mellan projekten listade nedan. Alternativt, om du har en egen idé så får vi komma överens om projektet är lämpligt.

Redovisning är inte obligatoriskt, men programmet du lämnar in måste fungera och möta kraven innan du kan få godkänt.
Väljer du att redovisa programmet är det lättare för dig att visa hur och varför det fungerar, och därmed lättare för oss att betygsätta rättvist.

## Laborationsuppgifter

Välj ett av spelen nedan att implementera.

### Projekt 1: Hangman

Skapa en version av det klassiska spelet Hangman.

* Datorn väljer ett slumpmässigt ord från en fördefinierad lista av ord.

* Spelet visar hur många bokstäver ordet består av, men inte vilka bokstäver som är rätt.

* Spelaren ska gissa en bokstav i taget, och datorn ger feedback om bokstaven finns i ordet eller inte.

* Spelet fortsätter tills spelaren har gissat hela ordet eller har gjort tillräckligt många felaktiga gissningar.

### Projekt 2: Memory

Skapa en version av spelet Memory.

* Datorn väljer ett antal slumpmässiga siffror eller bokstäver (beroende på svårighetsgrad) och visar dem i en viss ordning.

* Sedan visar datorn samma siffror eller bokstäver igen, men denna gång blandat.

* Spelaren ska gissa i vilken ordning siffrorna eller bokstäverna visades första gången.

* Spelet fortsätter tills spelaren har gissat rätt ordning.

### Projekt 3: Sten-sax-påse

Skapa en version av spelet sten-sax-påse.

* Datorn slumpar vilken av sten, sax eller påse den ska välja.

* Spelaren väljer också sten, sax eller påse.

* Datorn och spelaren visar sedan upp sina val samtidigt.

* Reglerna är enligt följande: sten vinner över sax, sax vinner över påse, och påse vinner över sten. Om båda väljer samma alternativ blir det oavgjort.

* Spelaren spelar tills hen vinner eller förlorar mot datorn.

### Projekt 4: Black Jack

Skapa ett program som simulerar ett blackjack-spel mellan en spelare och en dator.

* Spelet spelas med en vanlig kortlek som blandas innan varje runda.

* Varje spelare får två kort i början av spelet. Datorn visar bara upp ett av sina kort.

* Spelaren kan välja att ta fler kort (hit) eller stanna på sina nuvarande kort (stand).

* Spelaren kan fortsätta att ta kort tills hen når 21 poäng eller över.

* Om spelaren går över 21 poäng förlorar hen direkt.

* När spelaren stannar, spelar datorn sin tur. Datorn måste ta kort så länge summan av korten är mindre än 17 poäng och stanna när datorns kortsumma är 17 poäng eller mer.

* Om datorn går över 21 poäng vinner spelaren oavsett vilka kort spelaren har.

* Om varken spelaren eller datorn går över 21 poäng så vinner den som har högst kortsumma.

## Krav på projekt

* Lösningen ska innehålla minst en klass med lämpliga metoder och attribut
* Lösningen ska innehålla minst en funktion (utöver klass-metoderna). Kan räcka med en main()-funktion
* Det ska gå att köra programmet
* Programmet ska fungera korrekt

## Inlämning av projektet

Skapa ett nytt repository på GitHub för ditt projekt.
Implementera projektet enligt dess beskrivning.
Lägg till en README-fil med instruktioner för hur man använder programmet eller spelar spelet.
Lägg eventuellt till en .gitignore-fil för att exkludera onödiga filer från versionhantering. Förslagsvis innehållande namnet på ditt virtual environement exempelvis (heter det venv har du venv/ på en rad i .gitignore)
Pusha alla filer till ditt repository på GitHub.
Lämna in en länk till ditt repository.
Om ditt repo är privat, lägg till <cj.freme@gmail.com> (användarnamn AladdinConslut) eller (<nibrasnubaid@gmail.com>) som collaborator genom att gå till "Settings" -> "Manage access" -> "Invite teams or people" -> skriv in användarnamnet AladdinConsult och välj "Collaborator" från rullgardinsmenyn.
