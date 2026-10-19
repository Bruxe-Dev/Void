#ifndef VOID_VAULT_ENCRYPTION_HPP
#define VOID_VAULT_ENCRYPTION_HPP

#include <string>

class Encryption
{
public:
    Encryption();

    bool encrypt(
        const std::string &plaintext,
        const std::string &password,
        std::string &encrypted_data);

    bool decrypt(
        const std::string &encrypted_data,
        const std::string &password,
        std::string &plaintext);
};

#endif
