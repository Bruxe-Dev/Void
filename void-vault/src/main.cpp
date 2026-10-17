
#include <iostream>

#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << std::boolalpha;

    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    bool unlocked = vault.unlock();

    std::cout << "Unlock successful: "
              << unlocked << '\n';

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    return 0;
}