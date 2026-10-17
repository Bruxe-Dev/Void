cmake_minimum_required(VERSION 3.15)

project(
    VoidVault
    DESCRIPTION "Local Configuration and Credential Protection Agent"
    VERSION 0.1.0
    LANGUAGES CXX 
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(
    VOID_VAULT_BUILD_TESTS
    "Build local vault tests"
    ON
)

option(
    VOID_VAULT_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

add_library(void_vault_options INTERFACE)

target_compile_features(
    void_vault_options
    INTERFACE cxx_std_20
)