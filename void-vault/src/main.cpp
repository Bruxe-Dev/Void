
#include <iostream>
#include <string>
#include "storage.hpp"

int main()
{
    Storage storage("storage_test.dat");
    std::string loaded;

    std::cout << "=== Storage Persistence Test ===\n";

    if (!storage.load(loaded))
    {
        std::cout << "Load: FAILED\n";
        return 1;
    }

    std::cout << "Load: SUCCESS\n";
    std::cout << "Saved data: " << loaded << '\n';

    if (loaded == "Void Vault storage test")
    {
        std::cout << "Persistence verification: PASSED\n";
    }
    else
    {
        std::cout << "Persistence verification: FAILED\n";
        return 1;
    }

    return 0;
}
