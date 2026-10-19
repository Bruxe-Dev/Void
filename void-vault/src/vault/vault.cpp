#include "vault.hpp"

#include <sodium.h>
#include <utility>

Vault::Vault(Authorizer& authorizer_ref, Storage& storage_ref)
    : authorizer(authorizer_ref),
      storage(storage_ref),
      is_unlocked(false)
{
}

Vault::~Vault() {
    lock();
}

bool Vault::isUnlocked() const {
    return is_unlocked;
}

bool Vault::authenticate(const std::string& attempted_password) {
    lock();

    if (!authorizer.verifyPassword(attempted_password)) {
        return false;
    }

    session_password = attempted_password;
    return true;
}

bool Vault::unlock() {
    if (!authorizer.isAuthorized() || session_password.empty()) {
        is_unlocked = false;
        return false;
    }

    is_unlocked = true;
    return true;
}

void Vault::lock() {
    is_unlocked = false;
    authorizer.revokeAuthorization();

    if (!session_password.empty()) {
        sodium_memzero(
            session_password.data(),
            session_password.size()
        );

        session_password.clear();
    }
}

bool Vault::saveData(const std::string& data) {
    if (!is_unlocked || session_password.empty()) {
        return false;
    }

    std::string encrypted_data;

    // Encrypt before writing anything to disk.
    if (!encryption.encrypt(
            data,
            session_password,
            encrypted_data
        )) {
        return false;
    }

    return storage.save(encrypted_data);
}

bool Vault::loadData(std::string& data) {
    if (!is_unlocked || session_password.empty()) {
        return false;
    }

    std::string encrypted_data;

    if (!storage.load(encrypted_data)) {
        return false;
    }

    std::string decrypted_data;

    if (!encryption.decrypt(
            encrypted_data,
            session_password,
            decrypted_data
        )) {
        return false;
    }

    data = std::move(decrypted_data);
    return true;
}

