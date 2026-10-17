#include "auth.hpp"

Authorizer::Authorizer()
    : authorized(false),
      password("void@2k26")
{
}

bool Authorizer::verifyPassword(
	const std:: string& attempted_password
)const{
	return attempted_password == password;
}
bool Authorizer::isAuthorized() const
{
    return authorized;
}
