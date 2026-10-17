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

if(MSVC)
    target_compile_options(void_vault_options INTERFACE 
        /w4
        /permissive-
        /sdl
        /GS
    )
else()
    target_compile_options(void_vault_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wformat=2
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            void_vault_options
            INTERFACE 
            -fstack-protector-strong
        )

    endif()
endif()

if(WIN32)
    if(MSVC)
        target_compile_options(
            void_vault_options
            INTERFACE /guard:cf
        )
        target_link_options(
            void_vault_options
            INTERFACE /guard:cf
        )

    endif()

elseif(UNIX AND NOT APPLE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(
            void_vault_options
            INTERFACE
            -wl,-z,relro,-z,now
        )
    endif()
endif()

if(VOID_VAULT_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
       AND NOT WIN32)
        target_compile_options(
            void_vault_options
            INTERFACE 
            -fsanitize-address,undefined
            -fno-omit-frame-pointer
        )

        target_link_options(
            void_vault_options
            INTERFACE 
            -fsanitize-address,undefined
        )
    
    else()
        message(FATAL_ERROR
            "Requested sanitizer configuration is not "
            "supported by this compiler/platform setup."
        )
    endif()
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/src/main.cpp")
    add_executable(void_vault
        src/main.cpp
    )

    target_link_libraries(
        void_vault
        PRIVATE 
        void_vault_options
    )

    target_include_directories(
        void_vault
        PRIVATE 
        "s{PROJECT_SOURCE_DIR}/include"
    )

else()
    message(STATUS
        "Application target deferred: src/main.cpp "
        "has not been created yet."
    )

endif()

if(VOID_VAULT_BUILD_TESTS)
    enable_testing()

    if(EXISTS "${PROJECT_SOURCE_DIR/tests/CMakeLists.txt}")
        add_subdirectory(tests)
    else()
        message(STATUS
            "Test targets deferred: tests/CMakeLists.txt "
            "has not been created yet."
        )
    endif()
endif()