#include <iostream>
#include <string>

#include "../src/security/auth.hpp"

int main()
{
    Authorizer authorizer;

    std::cout << "=== Authentication Tests ===\n";

    const bool result =
        authorizer.setPassword("SecurePassword123!");

    if (result)
    {
        std::cout << "PASS: Password setup\n";
    }
    else
    {
        std::cerr << "FAIL: Password setup\n";
        return 1;
    }

    return 0;
}
