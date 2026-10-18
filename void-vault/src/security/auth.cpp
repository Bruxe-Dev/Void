#include <stdexcept>
#include "auth.hpp"

#include <sodium.h>

Authorizer::Authorizer()
	: authorized(false)
{
	if(sodium_init() < 0){
		throw std::runtime_error("Libsodium Initialisation Error");
	}
	
	const std::string initial_password = "void@2k26";
	
	char hash[crypto_pwhash_STRBYTES];
	
	if(cryptp_pwhash_str(
		hash,
		initial_password.c_str(),
		initial_password.size(),
		crypto_pwhash_OPSLIMIT_INTERACTIVE,
		crypto_pwhash_MEMLIMIT_INTERACTIVE
	) != 0){
		throw std::runtime_error("Password hash Generation Failed");
	}
	
	password_hash = hash;
}

bool Authorizer::verifyPassword(
	const std::string& attempted_password
){
	authorized = (
		crypto_pwhash_str_verify(
			password_hash.c_str(),
			attempted_password.c_str(),
			attempted_password.size()
		) == 0
	);
	return authorized;
}

bool Authorizer::isAuthorized() const {
	return authorized;
}

bool Authorizer::revokeAuthorization(){
	authorized = false;
}
