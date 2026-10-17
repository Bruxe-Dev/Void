#include "vault.hpp"

Vault::Vault(Authorizer& authorizer_ref)
    : authorizer(authorizer_ref),
      is_unlocked(false)
{
}

bool Vault::isUnlocked() const {
    return is_unlocked;
}

bool Vault::unlock(){
	if(!authorizer.isAuthorized()){
		return false;
	}
	is_unlocked = true;
	return true;
}

void Vault::lock(){
	is_unlocked = false;
	authorizer.revokeAuthorization();
}
