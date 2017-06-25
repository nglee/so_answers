int main()
{
    int x = 4;
    int y = 4;
    int z = 4;

    for (int i = 0; i < 100; i++) {
        int fourth_index = i / (x * y * z);
        int third_index = i % (x * y * z) / (x * y);
        int second_index = i % (x * y * z) % (x * y) / x;
        int first_index = i % (x * y * z) % (x * y) % x;

        printf("%d: (%d, %d, %d, %d)\n", i, first_index, second_index, third_index, fourth_index);
    }
}
