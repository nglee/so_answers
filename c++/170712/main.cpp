#include <iostream>
#include <bitset>
#include <climits>

// my answer
bool solve(std::bitset<32> bit_S, std::bitset<32> bit_X, int index, int must_provide_carry)
{
    // end condition
    if (index == -1)
        if (must_provide_carry == 1)
            return false;
        else
            return true;

    bool _bit_S = bit_S[index];
    bool _bit_X = bit_X[index];
    switch (must_provide_carry) {
    case 2: // don't care
        if (_bit_S ^ _bit_X)
            return solve(bit_S, bit_X, --index, 1);
        else
            return solve(bit_S, bit_X, --index, 0);
    case 0: // we must not provide carry from this bit
        if (_bit_S ^ _bit_X)
            if (_bit_X)
                return false;
            else
                return solve(bit_S, bit_X, --index, 1);
        else
            return solve(bit_S, bit_X, --index, 0);
    case 1: // we must provide carry from this bit
        if (_bit_S ^ _bit_X)
            return solve(bit_S, bit_X, --index, 1);
        else
            if (_bit_X)
                return false;
            else
                return solve(bit_S, bit_X, --index, 0);
    }
}

// solution for the accepted answer
bool solve2(int S, int X)
{
    if (S < X) return false;
    int Y = S - X;
    if (Y & 1) return false;
    if ((X & (Y / 2)) != 0) return false;
    return true;
}

int main()
{
    /*while (true) {
     int S, X;
     std::cout << "S : ";
     std::cin >> S;
     std::cout << "X : ";
     std::cin >> X;
     */
    int k = 1024;
    #pragma omp parallel for
    for (int S = 0; S < INT_MAX; S++)
        for (int X = 0; X <= S; X++) {
            if (S == k && (k *= 2))
                std::cout << "S : " << S << "\n";
            std::bitset<32> bit_S(S);
            std::bitset<32> bit_X(X);

            if (solve(bit_S, bit_X, 31, 2) != solve2(S, X)) {
                std::cout << "FALSE!!\n";
                std::cout << "S : " << bit_S << " (" << S << ")\n";
                std::cout << "X : " << bit_X << " (" << X << ")\n";
                std::cout << "Press any key..\n";
                int d;
                std::cin >> d;
            }
      //std::cout << "S : " << bit_S << " (" << S << ")\n";
      //std::cout << "X : " << bit_X << " (" << X << ")\n";
      //std::cout << solve(bit_S, bit_X, 7, 2) << std::endl;
      //std::cout << solve2(S, X) << std::endl;
     }
}
