#ifndef VOID_VAULT_AUTHORIZER_HPP
#define VOID_VAULT_AUTHORIZER_HPP

class Authorizer {
public:
    Authorizer();

    bool isAuthorized() const;

private:
    bool authorized;
};

#endif

