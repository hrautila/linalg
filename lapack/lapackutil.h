
#ifndef LAPACK_UTIL_H
#define LAPACK_UTIL_H

extern void dlarf_(char *side, int *M, int *N, double *V, int *incv, double *tau, double *C, int *ldc, double *work);

extern void dlarfb_(char *side, char *trans, char *direct, char *storev, int *M, int *N, int *K, double *V, int *ldv, double *T, int *ldt, double *C, int *ldc, double *work, int *ldwork);

extern void dlarfx_(char *side, int *M, int *N, double *V, double *tau, double *C, int *ldc, double *work);

extern void dlarfg_(int *N, double *alpha, double *X, int *incx, double *tau);

extern void dlarft_(char *direct, char *storev, int *N, int *K, double *V, int *ldv, double *tau, double *T, int *ldt);

extern void dorgqr_(int *M, int *N, int *K, double *A, int *lda, double *tau, double *work, int *lwork, int *info);


#endif
