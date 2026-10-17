#include "auth.hpp"

Authorizer::Authorizer()
    : authorized(false),
      password("prototype-password")
{
}


bool Authorizer::verifyPassword(
    const std::string& attempted_password
) {
    authorized = (attempted_password == password);
    return authorized;
}

bool Authorizer::isAuthorized() const {
    return authorized;
}

void Authorizer::revokeAuthorization(){
	authprized = false;
}
