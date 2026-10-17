#ifndef VOID_VAULT_AUTH_HPP
#define VOID_VAULT_AUTH_HPP

class Authorizer {
public:
    Authorizer();

    bool isAuthorized() const;

private:
    bool authorized;
};

#endif

