#include <iostream>

#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << std::boolalpha;

    std::cout << "=== Local Vault ===\n";

    std::cout << "\n1. Initial state\n";
    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';
    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n2. Attempt unlock before authentication\n";
    std::cout << "Unlock successful: "
              << vault.unlock() << '\n';

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n3. Authenticate\n";
    std::cout << "Authentication successful: "
              << authorizer.verifyPassword("prototype-password")
              << '\n';

    std::cout << "\n4. Attempt unlock after authentication\n";
    std::cout << "Unlock successful: "
              << vault.unlock() << '\n';

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n5. Lock the vault\n";
    vault.lock();

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    return 0;
}
