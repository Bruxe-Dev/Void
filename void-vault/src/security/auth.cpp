#include "auth.hpp"

Authorizer::Authorizer()
    : authorized(false)
{
}

bool Authorizer::isAuthorized() const
{
    return authorized;
}