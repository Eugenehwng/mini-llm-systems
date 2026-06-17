#include <iostream>

void add_three(int &a) { a = a + 3; }

int main() {
    int a = 10;
    int &b = a;

    std::cout << b << " | " << a << std::endl;

    add_three(a);
    std::cout << a << std::endl;
}