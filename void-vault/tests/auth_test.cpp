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
    const bool short_password_result =
        authorizer.setPassword("short");

    if (!short_password_result)
    {
        std::cout << "PASS: Reject short password\n";
    }
    else
    {
        std::cerr << "FAIL: Accepted short password\n";
        return 1;
    }
    return 0;
}
