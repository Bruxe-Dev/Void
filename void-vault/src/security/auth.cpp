#include "auth.hpp"

#include <sodium.h>
#include <stdexcept>

Authorizer::Authorizer()
    : authorized(false)
{
    if (sodium_init() < 0) {
        throw std::runtime_error("Libsodium initialization failed");
    }
}


bool Authorizer::setPassword(const std::string& new_password) {
    if (new_password.size() < 12) {
        return false;
    }

    char hash[crypto_pwhash_STRBYTES];

    if (crypto_pwhash_str(
            hash,
            new_password.c_str(),
            new_password.size(),
            crypto_pwhash_OPSLIMIT_INTERACTIVE,
            crypto_pwhash_MEMLIMIT_INTERACTIVE
        ) != 0) {
        return false;
    }

    password_hash = hash;
    authorized = false;
    return true;
}


bool Authorizer::verifyPassword(
    const std::string& attempted_password
) {
    if (locked_out || password_hash.empty()) {
        authorized = false;
        return false;
    }

    if (crypto_pwhash_str_verify(
            password_hash.c_str(),
            attempted_password.c_str(),
            attempted_password.size()
        ) == 0) {
        failed_attempts = 0;
        authorized = true;
        return true;
    }

    authorized = false;
    ++failed_attempts;

    if (failed_attempts >= 5) {
        locked_out = true;
    }

    return false;
}


bool Authorizer::isAuthorized() const {
    return authorized;
}

void Authorizer::revokeAuthorization() {
    authorized = false;
}

