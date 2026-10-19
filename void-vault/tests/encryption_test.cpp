#include "../src/security/encryption.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main()
{
    Encryption encryption;

    const std::string password = "TestPassword123!";
    const std::string original = "Void Vault secret data";

    std::string encrypted;
    std::string decrypted;

    assert(encryption.encrypt(original, password, encrypted));
    assert(encrypted != original);
    assert(encryption.decrypt(encrypted, password, decrypted));
    assert(decrypted == original);

    std::string result;
    assert(!encryption.decrypt(
        encrypted, "IncorrectPassword!", result));

    std::string modified = encrypted;
    modified.back() ^= 1;

    assert(!encryption.decrypt(modified, password, result));

    assert(!encryption.decrypt("VV01", password, result));

    std::cout << "All encryption tests passed.\n";
    return 0;
}