#include "test_framework.hpp"
#include <iostream>

int main() {
    auto& tests = ts_test::get_test_registry();
    std::cout << "Running " << tests.size() << " test cases..." << std::endl;

    size_t passed = 0;
    for (const auto& test : tests) {
        std::cout << "[ RUN      ] " << test.name << std::endl;
        test.func();
        std::cout << "[       OK ] " << test.name << std::endl;
        passed++;
    }

    std::cout << "[==========] " << passed << " tests passed." << std::endl;
    return 0;
}
