#ifndef VOID_VAULT_VAULT_HPP
#define VOID_VAULT_VAULT_HPP

#include <string>

#include "../security/auth.hpp"
#include "../security/encryption.hpp"
#include "../../include/storage.hpp"

class Vault {
public:
    Vault(Authorizer& authorizer, Storage& storage);
    ~Vault();

    bool isUnlocked() const;
    bool authenticate(const std::string& attempted_password);
    bool unlock();
    void lock();

    bool saveData(const std::string& data);
    bool loadData(std::string& data);

private:
    Authorizer& authorizer;
    Storage& storage;
    Encryption encryption;
    std::string session_password;
    bool is_unlocked;
};

#endif

