#include "auth.hpp"

Authorizer::Authorizer()
    : authorized(false),
      password("void@2k26")
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

