
#include <iostream>

#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << "Authorized: "
              << std::boolalpha
              << authorizer.isAuthorized() << '\n';

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    return 0;
}
