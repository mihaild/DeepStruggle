#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cstdlib>

namespace ts_test {

struct TestCase {
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestCase>& get_test_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

inline void register_test(const std::string& name, std::function<void()> func) {
    get_test_registry().push_back({name, func});
}

} // namespace ts_test

#define TEST(suite, name) \
    void test_##suite##_##name(); \
    struct Register_##suite##_##name { \
        Register_##suite##_##name() { ::ts_test::register_test(#suite "." #name, test_##suite##_##name); } \
    } reg_##suite##_##name; \
    void test_##suite##_##name()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define ASSERT_FALSE(cond) \
    do { \
        if (cond) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": NOT(" #cond ")" << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        if (!((a) == (b))) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": (" #a " == " #b ")" << std::endl; \
            std::exit(1); \
        } \
    } while(0)
