#include "vault.hpp"

Vault::Vault(Authorizer& authorizer_ref)
    : authorizer(authorizer_ref),
      is_unlocked(false)
{
}

bool Vault::isUnlocked() const {
    return is_unlocked;
}

void Vault::lock(){
	is_unlocked = false;
}
