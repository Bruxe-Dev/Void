#include <iostream>

#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << std::boolalpha;

    std::cout << "=== Void Vault ===\n";

    std::cout << "\n1. Initial state\n";
    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';
    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n2. Unlock before authentication\n";
    std::cout << "Unlock successful: "
              << vault.unlock() << '\n';
    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n3. Authenticate with correct password\n";
    bool authenticated =
        vault.authenticate("prototype-password");

    std::cout << "Authentication successful: "
              << authenticated << '\n';
    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';

    std::cout << "\n4. Unlock after authentication\n";
    bool unlocked = vault.unlock();

    std::cout << "Unlock successful: "
              << unlocked << '\n';
    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n5. Attempt incorrect password\n";
    bool wrong_attempt =
        vault.authenticate("wrong-password");

    std::cout << "Authentication successful: "
              << wrong_attempt << '\n';
    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';
    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';

    std::cout << "\n6. Lock the vault\n";
    vault.lock();

    std::cout << "Vault unlocked: "
              << vault.isUnlocked() << '\n';
    std::cout << "Authorized: "
              << authorizer.isAuthorized() << '\n';

    return 0;
}
