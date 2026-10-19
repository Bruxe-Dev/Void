add_executable(storage_test
    storage_test.cpp
    ../src/storage/storage.cpp
)

target_link_libraries(
    storage_test
    PRIVATE void_vault_options
)

target_include_directories(
    storage_test
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
)

add_test(
    NAME storage_test
    COMMAND storage_test
)

add_executable(auth_test
	auth_test.cpp
	../src/security/auth.cpp
)

target_link_libraries(
    auth_test
    PRIVATE
    void_vault_options
    PkgConfig::SODIUM
)

target_include_directories(
	auth_test
	PRIVATE "${PROJECT_SOURCE_DIR}src/security" 
)

add_test(
	NAME auth_test
	COMMAND auth_test
)

add_executable(encryption_test
    encryption_test.cpp
    ../src/security/encryption.cpp
)

target_link_libraries(
    encryption_test
    PRIVATE
    void_vault_options
    PkgConfig::SODIUM
)

target_include_directories(
    encryption_test
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
)

add_test(
    NAME encryption_test
    COMMAND encryption_test
)