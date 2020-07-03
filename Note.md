## Main Idea
- avoid condition branch
when compiler comples the code, pipeline will help excute instructions into fewer clock cycle. While it encouters conditional branch, compiler cannot preload instructions (it may guess which branch to go and preload).

### MMult_0
#### [MMult0.c](src/HowToOptimizeGemm/MMult0.c)
basic implementation of matrix multiplication

`C(i,j) = row_vec_A(i,0) * column_vec_B(0,j)`

### MMult_1
#### [MMult1.c](src/HowToOptimizeGemm/MMult1.c)
abstract vector multiplication -> AddDot1x1

Use of macros: note here incx passed an arg but used by macros,
since `*gamma += X( p ) * y[ p ];` is replaced by `*gamma += x[ (i)*incx ] * y[ p ]`;         
```c
#define X(i) x[ (i)*incx ]

void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  int p;

  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];     
  }
}
```
### MMult_1x4
#### MMult_1x4_3.c(src/MMult_1x4_3.c)
now compute 4 C elements (on same row) at a time -> AddDot1x4 = 4*AddDot1x1

TODO: would this only increase the efficiency?
#### MMult_1x4_4.c(src/MMult_1x4_5.c)
make AddDot inline

TODO: why inline?
#### MMult_1x4_5.c(src/MMult_1x4_4.c)
put inline AddDot in one loop

TODO: why works? Increase cache hit rates or helps compiler to parrell?

#### MMult_1x4_6.c(src/MMult_1x4_4.c)
put C and A to registers

TODO: what is the valid number of registers could be used in C? Or in compiler?
Maybe related to how many cores?

#### MMult_1x4_7.c(src/MMult_1x4_4.c)
put pointer of B into register `b_p0_pntr = &B( p, 0 );`
now compute c as `c_00_reg += a_0p_reg * *b_p0_pntr;`

### MMult_4x4
###  MMult_4x4_3 -> MMult_1x4_4_7
Same as mmult_1x4, but change to 4x4 block. AddDot4x4 = 4*AddDot1x4 = 4\*AddDot1x1
#### MMult_1x4_8.c(src/MMult_1x4_4.c)
put element of B into register `b_p0_reg = *b_p0_pntr++;`
now compute c as `c_00_reg += a_0p_reg * b_p0_reg`

#### MMult_1x4_9.c(src/MMult_1x4_4.c)
re-arrange the code for next version.

#### MMult_1x4_10.c(src/MMult_1x4_4.c)
introduce vector registers.
```c
// as union, v and d[2] share the same memory 
typedef union 
{
  __m128d v; // 128 bits -> 16 bytes -> 2 double
  double d[2];
} v2df_t;

// c_00_c_10_vreg, v2df_t.d[0] = C(0,0), v2df_t.d[1] = C(1,0) (double)
    c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg, 
// a_0p_a1p_vreg, v2df_t.d[0] = &A(0,p), v2df_t.d[1] = &A(1,p) (double*)
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
// b_p0_vreg, v2df_t.d[0] = &B(0, p), v2df_t.d[1] = &B(0, p)
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;
```

As mentioned in step-by-step note:_SSE3 instruction set that allow one to perform two 'multiply accumulate' operations (two multiplies and two adds) per clock cycle for a total of four floating point operations per clock cycle_. Now the following computation can finish in one clock cylcle.
```c
    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v; 
    // -> C(0, 0) += A(0, p) * B(p, 0)
    // -> C(1, 0) += A(1, p) * B(p, 0) 
```

### MMult_4x4_11.c(src/MMult_1x4_4.c)
Since MMult_4x4_10 version's performance will go down when matrix size gets larger. 
TODO: why? Ensure how block is inside the cache?

We further split matrix into small chunks. 

#### MMult_4x4_12.c(src/MMult_1x4_4.c)
After split, rows(A)' memory address is not continuous any more. So we need to pack it.

#### MMult_4x4_13.c(src/MMult_1x4_4.c)
TODO: why?
```c
a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &A( 0, p ) );
a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &A( 2, p ) );
```
```c
a_0p_a_1p_vreg.v = _mm_load_pd( (double *) a );
a_2p_a_3p_vreg.v = _mm_load_pd( (double *) ( a+2 ) );
a += 4;
```

### MMult_4x4_14.c MMult_4x4_15.c
pack colums(B).

