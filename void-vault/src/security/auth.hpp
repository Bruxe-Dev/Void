#ifndef VOID_VAULT_AUTH_HPP
#define VOID_VAULT_AUTH_HPP

#include <string>

class Authorizer {
public:
    Authorizer();

    bool setPassword(const std::string& new_password);
    bool verifyPassword(const std::string& attempted_password);
    bool isAuthorized() const;
    void revokeAuthorization();
    bool isLockedOut() const;

private:
    bool authorized;
    std::string password_hash;
    int failed_attempts;
    bool locked_out;
};

#endif

