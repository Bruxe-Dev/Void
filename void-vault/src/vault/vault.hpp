#ifndef VOID_VAULT_VAULT_HPP
#define VOID_VAULT_VAULT_HPP

#include <string>
#include <string_view>

#include "../security/auth.hpp"
#include "../security/encryption.hpp"
#include "../security/secure_buffer.hpp"
#include "../../include/storage.hpp"

enum class VaultError {
    None,
    Locked,             // not authenticated / not unlocked
    NoData,             // no data file yet
    DecryptFailed,      // wrong password or corrupted / tampered data file
    StorageError,       // file could not be read or written
    EncryptFailed,
    ForeignDataExists   // existing data cannot be decrypted with this password; refusing to overwrite
};

const char* vaultErrorMessage(VaultError error);

class Vault {
public:
    Vault(Authorizer& authorizer, Storage& storage);
    ~Vault();

    Vault(const Vault&) = delete;
    Vault& operator=(const Vault&) = delete;

    bool isUnlocked() const;
    bool authenticate(std::string_view attempted_password);
    bool unlock();
    void lock();

    bool saveData(std::string_view data);
    bool loadData(std::string& data);

    VaultError lastError() const;

private:
    Authorizer& authorizer;
    Storage& storage;
    Encryption encryption;
    SecureBuffer session_password;
    bool is_unlocked;
    VaultError last_error;
};

#endif

