
#ifndef VOID_VAULT_VAULT_HPP
#define VOID_VAULT_VAULT_HPP

#include "../security/auth.hpp"

class Vault {
public:
    explicit Vault(Authorizer& authorizer);

    bool isUnlocked() const;
    void lock();

private:
    Authorizer& authorizer;
    bool is_unlocked;
};

#endif

