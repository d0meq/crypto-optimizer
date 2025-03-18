import math

def sito_erastotenesa(n):
    if n < 2:
        return []
    
    liczby = list(range(1, n + 1))
    liczby[0] = 0
    
    j = 2
    
    while j <= math.sqrt(n):
        if liczby[j - 1] != 0:
            for k in range(2, n // j + 1):
                liczby[j * k - 1] = 0
        j += 1
    
    return [x for x in liczby if x != 0]

print(sito_erastotenesa(10))