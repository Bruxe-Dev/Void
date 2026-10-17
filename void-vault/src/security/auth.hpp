#ifndef VOID_VAULT_AUTH_HPP
#define VOID_VAULT_AUTH_HPP

class Auth{
	public:
		Auth();
		
		bool isAuthorised() const;
	private: 
		bool is_authorised;
};

#endif
