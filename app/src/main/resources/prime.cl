__kernel void filterPrime(__global int* filteredPrimes, const int length) {
    int filter = get_global_id(0) * 2 + 3;
    for (int i = get_global_id(0) + filter; i < length; i += filter) {
        filteredPrimes[i] = 1;
    }
}
