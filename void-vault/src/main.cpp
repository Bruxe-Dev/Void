#include <iostream>
#include "security/auth.hpp"
#include "vault/vault.hpp"
#include "storage.hpp"

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

    std::cout << "\n3. Set and verify password\n";

    if (!authorizer.setPassword("PrototypePassword123!"))
    {
        std::cerr << "Password setup failed.\n";
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

    std::cout << "Locked out before test: "
              << (authorizer.isLockedOut() ? "true" : "false")
              << '\n';

    for (int i = 1; i <= 4; ++i)
    {
        bool result = vault.authenticate("WrongPassword123!");

        std::cout << "Additional attempt " << i
                  << ": " << (result ? "Success" : "Failed")
                  << " | Locked out: "
                  << (authorizer.isLockedOut() ? "true" : "false")
                  << '\n';
    }

    std::cout << "\nFinal lockout status: "
              << (authorizer.isLockedOut() ? "true" : "false")
              << '\n';

    std::cout << "\n8. Testing password-reset protection\n";

    bool reset =
        authorizer.setPassword("AnotherPrototypePassword123!");

    std::cout << "Second password setup accepted: "
              << (reset ? "true" : "false") << '\n';
    std::cout << "Still locked out: "
              << (authorizer.isLockedOut() ? "true" : "false")
              << '\n';

    std::cout << "\n=== End of tests ===\n";

    return 0;
}
