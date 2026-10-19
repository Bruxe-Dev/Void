#include "auth.hpp"

#include <sodium.h>

#include <fstream>
#include <string>

Authorizer::Authorizer()
    : authorized(false),
      failed_attempts(0),
      locked_out(false),
      password_initialized(false),
      persistent_storage(false),
      storage_error(false)
{
    if (sodium_init() < 0) {
        storage_error = true;
    }
}

Authorizer::Authorizer(const std::string& file_path)
    : authorized(false),
      failed_attempts(0),
      locked_out(false),
      password_initialized(false),
      hash_file_path(file_path),
      persistent_storage(true),
      storage_error(false)
{
    if (sodium_init() < 0) {
        storage_error = true;
        return;
    }

    if (!loadPasswordHash()) {
        storage_error = true;
    }
}

bool Authorizer::loadPasswordHash() {
    std::ifstream file(hash_file_path, std::ios::binary);

    if (!file.is_open()) {

        std::ifstream check(hash_file_path, std::ios::binary);
        if (check.good()) {
            return false;
        }

        password_initialized = false;
        return true;
    }

    password_initialized = true;

    std::string stored_hash;
    if (!std::getline(file, stored_hash)) {
        return false;
    }

    if (stored_hash.empty() ||
        stored_hash.find("$argon2id$") == std::string::npos) {
        return false;
    }

    password_hash = stored_hash;
    return true;
}

bool Authorizer::savePasswordHash(const std::string& hash) {
    if (!persistent_storage) {
        return true;
    }

    std::ofstream file(
        hash_file_path,
        std::ios::binary | std::ios::out | std::ios::trunc
    );

    if (!file.is_open()) {
        return false;
    }

    file.write(hash.data(), static_cast<std::streamsize>(hash.size()));
    file.put('\n');
    file.flush();

    return file.good();
}

bool Authorizer::setPassword(const std::string& new_password) {
    if (storage_error ||
        password_initialized ||
        new_password.size() < 12) {
        return false;
    }

    char generated_hash[crypto_pwhash_STRBYTES] = {};

    if (crypto_pwhash_str(
            generated_hash,
            new_password.c_str(),
            static_cast<unsigned long long>(new_password.size()),
            crypto_pwhash_OPSLIMIT_INTERACTIVE,
            crypto_pwhash_MEMLIMIT_INTERACTIVE
        ) != 0) {
        sodium_memzero(generated_hash, sizeof(generated_hash));
        return false;
    }

    const std::string candidate_hash(generated_hash);

    sodium_memzero(generated_hash, sizeof(generated_hash));

    if (!savePasswordHash(candidate_hash)) {
        return false;
    }

    password_hash = candidate_hash;
    password_initialized = true;
    failed_attempts = 0;
    locked_out = false;

    return true;
}

bool Authorizer::verifyPassword(
    const std::string& attempted_password
) {
    if (storage_error ||
        locked_out ||
        !password_initialized ||
        password_hash.empty()) {
        return false;
    }

    if (crypto_pwhash_str_verify(
            password_hash.c_str(),
            attempted_password.c_str(),
            static_cast<unsigned long long>(attempted_password.size())
        ) == 0) {
        authorized = true;
        failed_attempts = 0;
        return true;
    }

    ++failed_attempts;

    if (failed_attempts >= 5) {
        locked_out = true;
        authorized = false;
    }

    return false;
}

bool Authorizer::isAuthorized() const {
    return authorized;
}

void Authorizer::revokeAuthorization() {
    authorized = false;
}

bool Authorizer::isLockedOut() const {
    return locked_out;
}

