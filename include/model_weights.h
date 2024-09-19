#include <stdint.h>


// Weights and biases for dense
const int8_t dense_weights[] = {
0, 0, 
};
const int32_t dense_biases[] = {
-6, 1, -3, -4, 0, 0, -3, 0, 0, -1, 
0, 0, -5, -2, 0, 0, 2, -4, -2, 0, 
-1, -1, -2, -1, -5, 0, 0, -2, 0, 0, 
-4, -4, 1, 0, 0, -7, 0, 0, 0, 0, 
-5, -6, -4, -1, 0, 0, -2, 0, -2, 0, 
2, -5, 0, -5, -5, 0, -7, -1, 0, 0, 
1, -3, -6, -6, -7, 0, 0, 0, -4, 0, 
0, 0, -4, 0, 0, 0, 1, -4, -3, 0, 
-5, 0, 0, -3, 0, 0, -7, -4, -3, 0, 
0, 0, 0, 0, 0, -8, 0, -1, -7, -7, 
-5, 1, 0, -4, 0, 0, -3, 0, -6, -1, 
0, 0, -3, -1, -6, 2, -5, -6, 0, 0, 
-3, 0, 0, 0, -2, -3, -3, -8, -6, -4, 
0, -7, 0, -4, 0, -6, -5, 0, 0, 0, 
0, -3, 2, -3, 0, 0, 2, -1, -3, -1, 
0, -2, 0, 0, -6, -2, 0, -8, 0, -1, 
0, 0, -2, 2, 0, 0, -5, 1, 0, 0, 
0, -3, 2, 0, -7, -5, -3, 1, 0, -2, 
0, -7, 0, -5, 0, 0, -8, 0, 0, -1, 
-3, -5, 0, -5, 0, -1, 0, 2, -5, -1, 
1, 0, 0, 0, 0, -5, 1, 0, 0, -3, 
0, 0, -4, 0, -6, 0, 0, 0, 0, -6, 
0, 1, 0, -3, -6, -5, 0, -3, -5, -7, 
0, -5, -1, -7, 2, 0, 0, 0, 0, 0, 
-7, 0, -5, 0, 0, 0, 0, 0, -4, 0, 
0, -7, 0, -5, 0, 0, 
};

// Weights and biases for dense_1
const int8_t dense_1_weights[] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 
};
const int32_t dense_1_biases[] = {
-213, -24, 358, -4, -152, -163, -5, 109, 243, -92, 
184, 240, -106, -47, 48, -130, -147, 69, -8, -175, 
332, 9, 279, -139, -109, 319, -78, 218, -172, 54, 
200, -247, -13, -129, -232, 145, -244, 120, 216, -138, 
103, -245, -190, -172, -178, -41, -412, -120, 144, 177, 
139, -129, 349, 93, -130, -212, -142, 39, -108, -233, 
-80, 103, -183, 157, -130, -249, 133, -81, -89, -180, 
103, 61, -105, -140, -73, 103, 122, -374, 267, 10, 
-32, -31, -61, -213, -234, -125, -177, 80, -148, -21, 
-6, -78, -182, -61, 234, -18, 197, -141, -11, 217, 
227, -199, 258, 254, -136, -38, 215, -237, -117, -33, 
180, 303, 239, 104, -127, -225, -97, 95, 191, -12, 
250, -117, 28, 113, -163, 169, 216, 23, -91, -169, 
225, -127, -102, -222, 97, 82, -136, 267, 245, 180, 
156, -138, 305, 337, 127, 240, -142, -88, 206, -252, 
156, -182, -165, -78, -77, -132, -98, 17, -106, -47, 
-136, 132, 189, -111, -95, 249, 101, -94, 108, -148, 
250, -123, -26, -59, 3, -57, -214, 6, 44, 79, 
-132, -75, 66, 394, 87, 98, 157, -110, 118, -189, 
-115, -284, 117, -219, -76, -106, -239, -223, -79, -68, 
156, 311, -114, -236, 206, -85, -175, -69, -121, -34, 
262, -239, -136, -129, -251, -125, -93, 47, 86, -13, 
-168, 190, 196, -107, -111, 166, -3, -148, 171, -103, 
216, 30, 91, 144, 169, 142, 161, 91, -127, -56, 
-52, -103, 156, -133, 35, 25, -34, -6, 178, 186, 
196, 188, -151, -203, -215, -120, 
};

// Weights and biases for dense_2
const int8_t dense_2_weights[] = {
0, 0, 
};
const int32_t dense_2_biases[] = {
-281, 281, 
};

