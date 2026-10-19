#include "vault.hpp"

Vault::Vault(Authorizer& authorizer_ref, Storage& storage_ref)
    : authorizer(authorizer_ref),
      storage(storage_ref),
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

bool Vault::saveData(const std::string& data) {
    if (!is_unlocked) {
        return false;
    }

    return storage.save(data);
}

bool Vault::loadData(std::string& data) {
    if (!is_unlocked) {
        return false;
    }

    return storage.load(data);
}
