int main()
{
    int x = 4;
    int y = 4;
    int z = 4;

    for (int i = 0; i < 100; i++) {
        int first_index = i % x;
        int second_index = i / x % y;
        int third_index = i / x / y % z;
        int fourth_index = i / x / y / z;

        printf("%d: (%d, %d, %d, %d)\n", i, first_index, second_index, third_index, fourth_index);
    }
}
