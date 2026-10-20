#include "auth.hpp"

#include "../../include/storage.hpp"

#include <sodium.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>

namespace
{
    constexpr char FORMAT_HEADER[] = "VVAUTH1";
    constexpr char ARGON_PREFIX[] = "$argon2id$";

    bool validHash(const std::string &hash)
    {
        if (hash.empty() || hash.size() >= crypto_pwhash_STRBYTES ||
            hash.rfind(ARGON_PREFIX, 0) != 0)
        {
            return false;
        }
        // -1 means the string is not a valid hash for this algorithm.
        return crypto_pwhash_str_needs_rehash(
                   hash.c_str(),
                   crypto_pwhash_OPSLIMIT_INTERACTIVE,
                   crypto_pwhash_MEMLIMIT_INTERACTIVE) != -1;
    }

    bool parseCounter(const std::string &line, int &out)
    {
        constexpr char KEY[] = "failed=";
        if (line.rfind(KEY, 0) != 0)
        {
            return false;
        }
        const std::string digits = line.substr(sizeof(KEY) - 1);
        if (digits.empty() || digits.size() > 6)
        {
            return false;
        }
        int value = 0;
        for (char c : digits)
        {
            if (c < '0' || c > '9')
            {
                return false;
            }
            value = value * 10 + (c - '0');
        }
        out = value;
        return true;
    }
}

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
        password_hash.clear();
    }
}

bool Authorizer::loadPasswordHash() {
    std::error_code ec;
    const std::filesystem::path p(hash_file_path);
    const bool present = std::filesystem::exists(p, ec);
    if (ec) {
        return false;                    // cannot determine -> fail closed
    }
    if (!present) {
        password_initialized = false;    // genuinely new account
        return true;
    }

    // From here on an account record exists: any problem is an error,
    // never a silent re-initialization.
    password_initialized = true;

    std::string content;
    if (!Storage::readFile(hash_file_path, content) || content.empty()) {
        return false;
    }

    std::istringstream in(content);
    std::string first;
    if (!std::getline(in, first)) {
        return false;
    }

    if (first == FORMAT_HEADER) {
        std::string hash, counter, extra;
        if (!std::getline(in, hash) || !std::getline(in, counter)) {
            return false;
        }
        if (std::getline(in, extra) && !extra.empty()) {
            return false;
        }
        int failed = 0;
        if (!validHash(hash) || !parseCounter(counter, failed)) {
            return false;
        }
        password_hash = hash;
        failed_attempts = failed;
    } else {
        // Legacy format: exactly one line containing the hash.
        std::string extra;
        if (std::getline(in, extra) && !extra.empty()) {
            return false;
        }
        if (!validHash(first)) {
            return false;
        }
        password_hash = first;
        failed_attempts = 0;
    }

    locked_out = failed_attempts >= MAX_FAILED_ATTEMPTS;
    return true;
}

bool Authorizer::persist(const std::string& hash, int failed, bool allow_replace) {
    if (!persistent_storage) {
        return true;
    }
    std::string record;
    record.reserve(hash.size() + 32);
    record += FORMAT_HEADER;
    record += '\n';
    record += hash;
    record += "\nfailed=";
    record += std::to_string(failed);
    record += '\n';
    return Storage::atomicWrite(hash_file_path, record, allow_replace);
}

bool Authorizer::setPassword(std::string_view new_password) {
    if (storage_error ||
        password_initialized ||
        new_password.size() < MIN_PASSWORD_LENGTH) {
        return false;
    }

    char generated_hash[crypto_pwhash_STRBYTES] = {};

    if (crypto_pwhash_str(
            generated_hash,
            new_password.data(),
            static_cast<unsigned long long>(new_password.size()),
            crypto_pwhash_OPSLIMIT_INTERACTIVE,
            crypto_pwhash_MEMLIMIT_INTERACTIVE
        ) != 0) {
        sodium_memzero(generated_hash, sizeof(generated_hash));
        return false;
    }

    const std::string candidate_hash(generated_hash);

    sodium_memzero(generated_hash, sizeof(generated_hash));

    // allow_replace = false: never overwrite an account created meanwhile.
    if (!persist(candidate_hash, 0, false)) {
        return false;
    }

    password_hash = candidate_hash;
    password_initialized = true;
    failed_attempts = 0;
    locked_out = false;

    return true;
}

bool Authorizer::verifyPassword(std::string_view attempted_password) {
    authorized = false;

    if (storage_error ||
        locked_out ||
        !password_initialized ||
        password_hash.empty()) {
        return false;
    }

    // Count the attempt BEFORE verifying (fail closed if it cannot be recorded).
    const int pending = failed_attempts + 1;
    if (!persist(password_hash, pending, true)) {
        locked_out = true;
        return false;
    }
    failed_attempts = pending;

    if (crypto_pwhash_str_verify(
            password_hash.c_str(),
            attempted_password.data(),
            static_cast<unsigned long long>(attempted_password.size())
        ) == 0) {
        authorized = true;
        failed_attempts = 0;
        // If this write fails the on-disk counter is merely more restrictive.
        persist(password_hash, 0, true);
        return true;
    }

    if (failed_attempts >= MAX_FAILED_ATTEMPTS) {
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

bool Authorizer::isLockedOut() const {
    return locked_out;
}

bool Authorizer::isInitialized() const {
    return password_initialized && !storage_error;
}

bool Authorizer::hasStorageError() const {
    return storage_error;
}

int Authorizer::failedAttempts() const {
    return failed_attempts;
}
