#include <iostream>
#include "vault/vault.hpp"

int main()
{
    Vault vault;
    std::cout << "Vault Unlocked: "
              << std::boolalpha
              << vault.is_unlocked();
    return 0;
}