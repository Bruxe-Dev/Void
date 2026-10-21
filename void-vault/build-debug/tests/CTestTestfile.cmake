# CMake generated Testfile for 
# Source directory: D:/Void/void-vault/tests
# Build directory: D:/Void/void-vault/build-debug/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(storage_test "D:/Void/void-vault/build-debug/tests/storage_test.exe")
set_tests_properties(storage_test PROPERTIES  WORKING_DIRECTORY "D:/Void/void-vault/build-debug/tests" _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;4;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
add_test(auth_test "D:/Void/void-vault/build-debug/tests/auth_test.exe")
set_tests_properties(auth_test PROPERTIES  WORKING_DIRECTORY "D:/Void/void-vault/build-debug/tests" _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;4;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
add_test(encryption_test "D:/Void/void-vault/build-debug/tests/encryption_test.exe")
set_tests_properties(encryption_test PROPERTIES  WORKING_DIRECTORY "D:/Void/void-vault/build-debug/tests" _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;4;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
add_test(vault_test "D:/Void/void-vault/build-debug/tests/vault_test.exe")
set_tests_properties(vault_test PROPERTIES  WORKING_DIRECTORY "D:/Void/void-vault/build-debug/tests" _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;4;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
add_test(broker_test "D:/Void/void-vault/build-debug/tests/broker_test.exe")
set_tests_properties(broker_test PROPERTIES  WORKING_DIRECTORY "D:/Void/void-vault/build-debug/tests" _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;4;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
add_test(app_e2e_test "C:/Program Files/CMake/bin/cmake.exe" "-DAPP=D:/Void/void-vault/build-debug/void_vault.exe" "-DWORK_DIR=D:/Void/void-vault/build-debug/tests/e2e_data" "-P" "D:/Void/void-vault/tests/app_e2e_test.cmake")
set_tests_properties(app_e2e_test PROPERTIES  _BACKTRACE_TRIPLES "D:/Void/void-vault/tests/CMakeLists.txt;10;add_test;D:/Void/void-vault/tests/CMakeLists.txt;0;")
