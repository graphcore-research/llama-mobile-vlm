// clang++ -std=c++20 -stdlib=libc++ -o build/demo_tbl demo_tbl.cpp -march=native
// ./build/demo_tbl

#include <arm_neon.h>
#include <iostream>

void demo_tbl() {
    {
        std::cout << "\n--- ARM64 vqtbl1q_u8 Demo ---" << std::endl;

        // Create a lookup table with 16 entries (4-bit indices)
        uint8_t table_data[16] = {10, 20,  30,  40,  50,  60,  70,  80,
                                  90, 100, 110, 120, 130, 140, 150, 160};
        uint8x16_t lut = vld1q_u8(table_data);

        // Create indices (4-bit values packed into bytes)
        uint8_t indices_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        uint8x16_t indices = vld1q_u8(indices_data);

        // Perform 4-bit table lookup
        uint8x16_t result = vqtbl1q_u8(lut, indices);

        // Print results
        uint8_t output[16];
        vst1q_u8(output, result);

        std::cout << "Lookup Table: ";
        for (int i = 0; i < 16; i++) {
            std::cout << static_cast<int>(table_data[i]) << " ";
        }
        std::cout << "\n";

        std::cout << "Indices:      ";
        for (int i = 0; i < 16; i++) {
            std::cout << static_cast<int>(indices_data[i]) << " ";
        }
        std::cout << "\n";

        std::cout << "Results:      ";
        for (int i = 0; i < 16; i++) {
            std::cout << static_cast<int>(output[i]) << " ";
        }
        std::cout << "\n";
    }
    {
        std::cout << "\n--- ARM64 vqtbl4q_u8 Demo ---" << std::endl;

        // Create 4 lookup tables, each with 16 entries (supports 64-entry table)
        uint8_t table_data[64] = {
            11, 12, 13, 14, 15, 16, 17, 18,  // table 0
            19, 20, 21, 22, 23, 24, 25, 26,  // table 0
            31, 32, 33, 34, 35, 36, 37, 38,  // table 1
            39, 40, 41, 42, 43, 44, 45, 46,  // table 1
            51, 52, 53, 54, 55, 56, 57, 58,  // table 2
            59, 60, 61, 62, 63, 64, 65, 66,  // table 2
            71, 72, 73, 74, 75, 76, 77, 78,  // table 3
            79, 80, 81, 82, 83, 84, 85, 86   // table 3
        };

        // Load 4 tables of 16 entries each
        uint8x16_t lut0 = vld1q_u8(&table_data[0]);
        uint8x16_t lut1 = vld1q_u8(&table_data[16]);
        uint8x16_t lut2 = vld1q_u8(&table_data[32]);
        uint8x16_t lut3 = vld1q_u8(&table_data[48]);

        // Create indices (6-bit values to index into 64-entry table)
        uint8_t indices_data[16] = {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51};
        uint8x16_t indices = vld1q_u8(indices_data);

        // Perform table lookup across 4 tables using vqtbl4q_u8
        uint8x16x4_t tables = {lut0, lut1, lut2, lut3};
        uint8x16_t result = vqtbl4q_u8(tables, indices);

        // Print results
        uint8_t output[16];
        vst1q_u8(output, result);

        std::cout << "Lookup Tables (64 entries total): ";
        for (int i = 0; i < 64; i++) {
            std::cout << static_cast<int>(table_data[i]) << " ";
        }
        std::cout << "\n";

        std::cout << "Indices (6-bit):                  ";
        for (int i = 0; i < 16; i++) {
            std::cout << static_cast<int>(indices_data[i]) << " ";
        }
        std::cout << "\n";

        std::cout << "Results:                          ";
        for (int i = 0; i < 16; i++) {
            std::cout << static_cast<int>(output[i]) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    demo_tbl();
    return 0;
}
