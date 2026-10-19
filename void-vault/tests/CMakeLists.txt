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
