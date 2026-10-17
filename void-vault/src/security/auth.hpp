#ifndef VOID_VAULT_AUTH_HPP
#define VOID_VAULT_AUTH_HPP

#include <string>

class Authorizer {
public:
    Authorizer();

	bool verifyPassword(const std::string& attempted_password) const;
    bool isAuthorized() const;

private:
    bool authorized;
    std::string password;
};

#endif

