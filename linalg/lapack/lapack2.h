/*
 * These prototypes are from CVXOPT source file lapack.c
 */

#ifndef LAPACK_H
#define LAPACK_H

/* LAPACK prototypes */
extern int ilaenv_(int  *ispec, char **name, char **opts, int *n1, int *n2, int *n3, int *n4);

extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb);
extern void zlacpy_(char *uplo, int *m, int *n, complex *A, int *lda, complex *B, int *ldb);

extern void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
extern void zgetrf_(int *m, int *n, complex *A, int *lda, int *ipiv, int *info);
extern void dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
extern void zgetrs_(char *trans, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
extern void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern void zgetri_(int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
extern void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
extern void zgesv_(int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);

extern void dgbtrf_(int *m, int *n, int *kl, int *ku, double *AB, int *ldab, int *ipiv, int *info);
extern void zgbtrf_(int *m, int *n, int *kl, int *ku, complex *AB, int *ldab, int *ipiv, int *info);
extern void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs, double *AB, int *ldab, int *ipiv, double *B, int *ldB, int *info);
extern void zgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs, complex *AB, int *ldab, int *ipiv, complex *B, int *ldB, int *info);
extern void dgbsv_(int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *b, int *ldb, int *info);
extern void zgbsv_(int *n, int *kl, int *ku, int *nrhs, complex *ab, int *ldab, int *ipiv, complex *b, int *ldb, int *info);

extern void dgttrf_(int *n, double *dl, double *d, double *du, double *du2, int *ipiv, int *info);
extern void zgttrf_(int *n, complex *dl, complex *d, complex *du, complex *du2, int *ipiv, int *info);
extern void dgttrs_(char *trans, int *n, int *nrhs, double *dl, double *d, double *du, double *du2, int *ipiv, double *B, int *ldB, int *info);
extern void zgttrs_(char *trans, int *n, int *nrhs, complex *dl, complex *d, complex *du, complex *du2, int *ipiv, complex *B, int *ldB, int *info);
extern void dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du, double *B, int *ldB, int *info);
extern void zgtsv_(int *n, int *nrhs, complex *dl, complex *d, complex *du, complex *B, int *ldB, int *info);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotrf_(char *uplo, int *n, complex *A, int *lda, int *info);
extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);
extern void zpotrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);
extern void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotri_(char *uplo, int *n, complex *A, int *lda, int *info);
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);
extern void zposv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);

extern void dpbtrf_(char *uplo, int *n, int *kd, double *AB, int *ldab, int *info);
extern void zpbtrf_(char *uplo, int *n, int *kd, complex *AB, int *ldab, int *info);
extern void dpbtrs_(char *uplo, int *n, int *kd, int *nrhs, double *AB, int *ldab, double *B, int *ldb, int *info);
extern void zpbtrs_(char *uplo, int *n, int *kd, int *nrhs, complex *AB, int *ldab, complex *B, int *ldb, int *info);
extern void dpbsv_(char *uplo, int *n, int *kd, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);
extern void zpbsv_(char *uplo, int *n, int *kd, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);

extern void dpttrf_(int *n, double *d, double *e, int *info);
extern void zpttrf_(int *n, double *d, complex *e, int *info);
extern void dpttrs_(int *n, int *nrhs, double *d, double *e, double *B, int *ldB, int *info);
extern void zpttrs_(char *uplo, int *n, int *nrhs, double *d, complex *e, complex *B, int *ldB, int *info);
extern void dptsv_(int *n, int *nrhs, double *d, double *e, double *B, int *ldB, int *info);
extern void zptsv_(int *n, int *nrhs, double *d, complex *e, complex *B, int *ldB, int *info);

extern void dsytrf_(char *uplo, int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern void zsytrf_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
extern void zhetrf_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
extern void dsytrs_(char *uplo, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
extern void zsytrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
extern void zhetrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
extern void dsytri_(char *uplo, int *n, double *A, int *lda, int *ipiv, double *work, int *info);
extern void zsytri_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *info);
extern void zhetri_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *info);
extern void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, double *work, int *lwork, int *info);
extern void zsysv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, complex *work, int *lwork, int *info);
extern void zhesv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, complex *work, int *lwork, int *info);

extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double  *a, int *lda, double *b, int *ldb, int *info);
extern void ztrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, complex  *a, int *lda, complex *b, int *ldb, int *info);
extern void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda, int *info);
extern void ztrtri_(char *uplo, char *diag, int *n, complex  *a, int *lda, int *info);
extern void dtbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd, int *nrhs, double *ab, int *ldab, double *b, int *ldb, int *info);
extern void ztbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd, int *nrhs, complex *ab, int *ldab, complex *b, int *ldb, int *info);

extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
extern void zgels_(char *trans, int *m, int *n, int *nrhs, complex *a, int *lda, complex *b, int *ldb, complex *work, int *lwork, int *info);
extern void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern void zgeqrf_(int *m, int *n, complex *a, int *lda, complex *tau, complex *work, int *lwork, int *info);
extern void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern void zunmqr_(char *side, char *trans, int *m, int *n, int *k, complex *a, int *lda, complex *tau, complex *c, int *ldc, complex *work, int *lwork, int *info);
extern void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
extern void zungqr_(int *m, int *n, int *k, complex *A, int *lda, complex *tau, complex *work, int *lwork, int *info);
extern void dorglq_(int *m, int *n, int *k, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
extern void zunglq_(int *m, int *n, int *k, complex *A, int *lda, complex *tau, complex *work, int *lwork, int *info);

extern void dgelqf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern void zgelqf_(int *m, int *n, complex *a, int *lda, complex *tau, complex *work, int *lwork, int *info);
extern void dormlq_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern void zunmlq_(char *side, char *trans, int *m, int *n, int *k, complex *a, int *lda, complex *tau, complex *c, int *ldc, complex *work, int *lwork, int *info);

extern void dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt, double *tau, double *work, int *lwork, int *info);
extern void zgeqp3_(int *m, int *n, complex *a, int *lda, int *jpvt, complex *tau, complex *work, int *lwork, double *rwork, int *info);

extern void dsyev_(char *jobz, char *uplo, int *n, double *A, int *lda, double *W, double *work, int *lwork, int *info);
extern void zheev_(char *jobz, char *uplo, int *n, complex *A, int *lda, double *W, complex *work, int *lwork, double *rwork, int *info);
extern void dsyevx_(char *jobz, char *range, char *uplo, int *n, double *A, int *lda, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, double *Z, int *ldz, double *work, int *lwork, int *iwork, int *ifail, int *info);
extern void zheevx_(char *jobz, char *range, char *uplo, int *n, complex *A, int *lda, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, complex *Z, int *ldz, complex *work, int *lwork, double *rwork, int *iwork, int *ifail, int *info);
extern void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA, double *W, double *work, int *lwork, int *iwork, int *liwork, int *info);
extern void zheevd_(char *jobz, char *uplo, int *n, complex *A, int *ldA, double *W, complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A, int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, double *Z, int *ldZ, int *isuppz, double *work, int *lwork, int *iwork, int *liwork, int *info);
extern void zheevr_(char *jobz, char *range, char *uplo, int *n, complex *A, int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, complex *Z, int *ldZ, int *isuppz, complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);

extern void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *A, int *lda, double *B, int *ldb, double *W, double *work, int *lwork,  int *info);
extern void zhegv_(int *itype, char *jobz, char *uplo, int *n, complex *A, int *lda, complex *B, int *ldb, double *W, complex *work, int *lwork, double *rwork, int *info);

extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A, int *ldA, double *S, double *U, int *ldU, double *Vt, int *ldVt, double *work, int *lwork, int *info);
extern void dgesdd_(char *jobz, int *m, int *n, double *A, int *ldA, double *S, double *U, int *ldU, double *Vt, int *ldVt, double *work, int *lwork, int *iwork, int *info);
extern void zgesvd_(char *jobu, char *jobvt, int *m, int *n, complex *A,
    int *ldA, double *S, complex *U, int *ldU, complex *Vt, int *ldVt,  complex *work, int *lwork, double *rwork, int *info);
extern void zgesdd_(char *jobz, int *m, int *n, complex *A, int *ldA, double *S, complex *U, int *ldU, complex *Vt, int *ldVt, complex *work, int *lwork, double *rwork, int *iwork, int *info);

extern void dgees_(char *jobvs, char *sort, void *select, int *n, double *A, int *ldA, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, int *bwork, int *info);
extern void zgees_(char *jobvs, char *sort, void *select, int *n, complex *A, int *ldA, int *sdim, complex *w, complex *vs, int *ldvs, complex *work, int *lwork, complex *rwork, int *bwork, int *info);
extern void dgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg, int *n, double *A, int *ldA, double *B, int *ldB, int *sdim, double *alphar, double *alphai, double *beta, double *vsl, int *ldvsl, double *vsr, int *ldvsr, double *work, int *lwork, int *bwork, int *info);
extern void zgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg, int *n, complex *A, int *ldA, complex *B, int *ldB, int *sdim, complex *alpha, complex *beta, complex *vsl, int *ldvsl, complex *vsr, int *ldvsr, complex *work, int *lwork, double *rwork, int *bwork, int *info);

#endif
