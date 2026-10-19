#ifndef VOID_VAULT_AUTH_HPP
#define VOID_VAULT_AUTH_HPP

#include <string>

class Authorizer {
public:
    Authorizer();

    explicit Authorizer(const std::string& hash_file_path);

    bool setPassword(const std::string& new_password);
    bool verifyPassword(const std::string& attempted_password);

    bool isAuthorized() const;
    void revokeAuthorization();
    bool isLockedOut() const;

private:
    bool loadPasswordHash();
    bool savePasswordHash(const std::string& hash);

    bool authorized;
    std::string password_hash;
    int failed_attempts;
    bool locked_out;
    bool password_initialized;

    std::string hash_file_path;
    bool persistent_storage;
    bool storage_error;
};

#endif

