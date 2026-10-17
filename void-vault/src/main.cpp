#include <iostream>
#include "vault/vault.hpp"

int main()
{
    Vault vault;
    std::cout << "Vault Unlocked: "
              << std::boolalpha
              << vault.isUnlocked();
    return 0;
}