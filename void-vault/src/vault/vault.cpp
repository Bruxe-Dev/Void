#include "vault.hpp"

Vault::Vault(Authorizer& authorizer_ref)
    : authorizer(authorizer_ref),
      is_unlocked(false)
{
}

bool Vault::isUnlocked() const {
    return is_unlocked;
}

bool Vault::authenticate(const std::string& attempted_password) {
    if (!authorizer.verifyPassword(attempted_password)) {
        lock();
        return false;
    }

    return true;
}

bool Vault::unlock() {
    if (!authorizer.isAuthorized()) {
        is_unlocked = false;
        return false;
    }

    is_unlocked = true;
    return true;
}

void Vault::lock() {
    is_unlocked = false;
    authorizer.revokeAuthorization();
}

