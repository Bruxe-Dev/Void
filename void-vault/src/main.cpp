#include <iostream>
#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    std::cout << "=== Void Vault ===\n";

    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << "\n1. Initial state\n";
    std::cout << "Authorized: "
              << (authorizer.isAuthorized() ? "true" : "false")
              << '\n';
    std::cout << "Vault unlocked: "
              << (vault.isUnlocked() ? "true" : "false")
              << '\n';

    std::cout << "\n2. Unlock before authentication\n";
    std::cout << "Unlock successful: "
              << (vault.unlock() ? "true" : "false")
              << '\n';
    std::cout << "Vault unlocked: "
              << (vault.isUnlocked() ? "true" : "false")
              << '\n';

    std::cout << "\n3. Authenticate with correct password\n";

    if (!authorizer.setPassword("PrototypePassword123!"))
    {
        std::cerr << "Failed to set prototype password.\n";
        return 1;
    }

    bool authenticated =
        vault.authenticate("PrototypePassword123!");

    std::cout << "Authentication successful: "
              << (authenticated ? "true" : "false") << '\n';
    std::cout << "Authorized: "
              << (authorizer.isAuthorized() ? "true" : "false")
              << '\n';

    std::cout << "\n4. Unlock after authentication\n";
    std::cout << "Unlock successful: "
              << (vault.unlock() ? "true" : "false") << '\n';
    std::cout << "Vault unlocked: "
              << (vault.isUnlocked() ? "true" : "false") << '\n';

    std::cout << "\n5. Attempt incorrect password\n";

    bool wrong_auth =
        vault.authenticate("DefinitelyWrongPassword!");

    std::cout << "Authentication successful: "
              << (wrong_auth ? "true" : "false") << '\n';
    std::cout << "Authorized: "
              << (authorizer.isAuthorized() ? "true" : "false")
              << '\n';
    std::cout << "Vault unlocked: "
              << (vault.isUnlocked() ? "true" : "false") << '\n';

    std::cout << "\n6. Lock the vault\n";
    vault.lock();

    std::cout << "Vault unlocked: "
              << (vault.isUnlocked() ? "true" : "false") << '\n';
    std::cout << "Authorized: "
              << (authorizer.isAuthorized() ? "true" : "false")
              << '\n';

    std::cout << "\n7. Testing five failed attempts\n";

    bool reset =
        authorizer.setPassword("NewPrototypePassword123!");

    std::cout << "Password reset: "
              << (reset ? "true" : "false") << '\n';
    std::cout << "Locked out after reset: "
              << (authorizer.isLockedOut() ? "true" : "false")
              << '\n';

    if (!reset || authorizer.isLockedOut())
    {
        std::cerr << "Could not reset authentication state.\n";
        return 1;
    }

    for (int i = 1; i <= 5; ++i)
    {
        bool result = vault.authenticate("WrongPassword123!");

        std::cout << "Attempt " << i
                  << ": " << (result ? "Success" : "Failed")
                  << " | Locked out: "
                  << (authorizer.isLockedOut() ? "true" : "false")
                  << '\n';
    }

    std::cout << "\n=== End of tests ===\n";

    return 0;
}
