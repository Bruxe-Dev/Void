#include "vault.hpp"

Vault::Vault()
{
	: is_unlocked(false)
}

bool Vault::is_unlocked() const {
	return is_unlocked;
}
