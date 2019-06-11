// #include <stdio.h>
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
    printf("%d+%d=%d\n", A[i],B[i], C[i]); 
}	