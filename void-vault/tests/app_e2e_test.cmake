# End-to-end test: runs the real void_vault executable as SEPARATE processes.
# Usage: cmake -DAPP=<exe> -DWORK_DIR=<dir> -P app_e2e_test.cmake
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(PW "E2ePassword-123456")
set(PW2 "E2eRotatedPass-7890!")
set(SECRET "e2e-confidential-payload-42")

function(run_app name input expect_rc)
    file(WRITE "${WORK_DIR}/${name}.in" "${input}")
    execute_process(
        COMMAND "${APP}" --data-dir "${WORK_DIR}"
        INPUT_FILE "${WORK_DIR}/${name}.in"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE rc
        TIMEOUT 120)
    file(REMOVE "${WORK_DIR}/${name}.in")
    set(LAST_OUT "${out}${err}" PARENT_SCOPE)
    if(NOT rc EQUAL ${expect_rc})
        message(FATAL_ERROR "${name}: exit ${rc}, expected ${expect_rc}\n${out}${err}")
    endif()
endfunction()

function(expect_contains name text)
    string(FIND "${LAST_OUT}" "${text}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "${name}: output missing '${text}'\n${LAST_OUT}")
    endif()
endfunction()

# Run 1: first-time setup, unlock, save, quit.
run_app(run1 "${PW}\n${PW}\n2\n1\n${PW}\n2\n${SECRET}\n6\n" 0)
expect_contains(run1 "Account created.")
expect_contains(run1 "Rejected: vault is locked")
expect_contains(run1 "Saved (encrypted).")
string(FIND "${LAST_OUT}" "${PW}" leaked)
if(NOT leaked EQUAL -1)
    message(FATAL_ERROR "run1: password echoed to output")
endif()

file(READ "${WORK_DIR}/vault_data.dat" data_hex HEX)
string(HEX "${SECRET}" secret_hex)
string(TOLOWER "${secret_hex}" secret_hex)
string(FIND "${data_hex}" "${secret_hex}" plain_pos)
if(NOT plain_pos EQUAL -1)
    message(FATAL_ERROR "plaintext found in vault_data.dat")
endif()
file(READ "${WORK_DIR}/vault_auth.dat" auth_txt)
string(FIND "${auth_txt}" "${PW}" pw_pos)
if(NOT pw_pos EQUAL -1)
    message(FATAL_ERROR "password found in vault_auth.dat")
endif()

# Run 2: new process, unlock, show data, rotate password (option 7), EOF exit.
run_app(run2 "1\n${PW}\n3\n7\n${PW}\n${PW2}\n${PW2}\n" 0)
expect_contains(run2 "Vault unlocked.")
expect_contains(run2 "Stored data: ${SECRET}")
expect_contains(run2 "Password changed and data re-encrypted.")
expect_contains(run2 "Input closed. Locking and exiting.")

# Run 2b: new process, verify rotated password unlocks and loads re-encrypted secret.
run_app(run2b "1\n${PW2}\n3\n6\n" 0)
expect_contains(run2b "Vault unlocked.")
expect_contains(run2b "Stored data: ${SECRET}")

# Run 3: invalid menu input handled; wrong passwords -> lockout.
run_app(run3 "abc\n1\nbad-password-1\n1\nbad-password-2\n1\nbad-password-3\n1\nbad-password-4\n1\nbad-password-5\n" 4)
expect_contains(run3 "Unknown option.")
expect_contains(run3 "Account is locked")

# Run 4: lockout persists across processes, even with the correct password.
run_app(run4 "1\n${PW2}\n" 4)
expect_contains(run4 "Account is locked")

# Run 5: corrupted auth file -> refuse, never re-initialize.
file(WRITE "${WORK_DIR}/vault_auth.dat" "corrupted")
run_app(run5 "${PW}\n${PW}\n" 3)
expect_contains(run5 "unreadable or corrupted")
file(READ "${WORK_DIR}/vault_auth.dat" after)
if(NOT after STREQUAL "corrupted")
    message(FATAL_ERROR "corrupted auth file was modified")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
message(STATUS "app_e2e_test passed")
