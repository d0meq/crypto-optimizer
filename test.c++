#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    srand(time(0));

    int n, x, tablica[30];

    cout << "Podaj liczbe n (maksymalnie 30): ";
    cin >> n;

    if (n < 1 || n > 30) {
        cout << "Nieprawidlowa wartosc n!" << endl;
        return 1;
    }

    // Generowanie liczb
    cout << "Wygenerowane liczby: ";
    int min_wartosc = 81;  
    int licznikPodzielnych = 0;
    
    for (int i = 0; i < n; i++) {
        tablica[i] = rand() % 80 + 1;
        cout << tablica[i] << " ";  
        if (tablica[i] < min_wartosc) min_wartosc = tablica[i];
    }
    
    cout << "\nMinimalna wartosc: " << min_wartosc;

    // Wyszukiwanie liczb podzielnych przez 3
    cout << "\nLiczby podzielne przez 3: ";
    for (int i = 0; i < n; i++) {
        if (tablica[i] % 3 == 0) {
            cout << tablica[i] << " ";
            licznikPodzielnych++;
        }
    }
    
    cout << "\nIlosc liczb podzielnych przez 3: " << licznikPodzielnych;

    // Szukanie pozycji liczby x
    cout << "\nPodaj liczbe x: ";
    cin >> x; 

    for (int i = 0; i < n; i++) {
        if (tablica[i] == x) {
            cout << "Liczba " << x << " wystepuje na pozycji: " << i + 1 << endl;
            return 0;
        }
    }

    cout << "Liczba " << x << " nie wystepuje w tablicy." << endl;
    return 0;
}
