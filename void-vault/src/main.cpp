
#include <iostream>
#include <string>

#include "security/auth.hpp"
#include "vault/vault.hpp"

int main()
{
    Authorizer authorizer;
    Vault vault(authorizer);

    std::cout << std::boolalpha;

    std::cout << "Testing incorrect password...\n";
    bool wrong_result = authorizer.verifyPassword("wrong-password");

    std::cout << "Verification: " << wrong_result << '\n';
    std::cout << "Authorized: " << authorizer.isAuthorized() << '\n';

    std::cout << "\nTesting correct prototype password...\n";
    bool correct_result = authorizer.verifyPassword("prototype-password");

    std::cout << "Verification: " << correct_result << '\n';
    std::cout << "Authorized: " << authorizer.isAuthorized() << '\n';

    return 0;
}
