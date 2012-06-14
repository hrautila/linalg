/*
 * These prototypes are from CVXOPT source file lapack.c
 */

#ifndef LAPACK_H
#define LAPACK_H

/* LAPACK prototypes */
extern int ilaenv_(int  *ispec, char **name, char **opts, int *n1,
    int *n2, int *n3, int *n4);

extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda,
    double *B, int *ldb);
extern void zlacpy_(char *uplo, int *m, int *n, void *A, int *lda,
    void *B, int *ldb);

extern void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv,
    int *info);
extern void zgetrf_(int *m, int *n, void *A, int *lda, int *ipiv,
    int *info);
extern void dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, int *info);
extern void zgetrs_(char *trans, int *n, int *nrhs, void *A, int *lda,
    int *ipiv, void *B, int *ldb, int *info);
extern void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work,
    int *lwork, int *info);
extern void zgetri_(int *n, void *A, int *lda, int *ipiv, void *work,
    int *lwork, int *info);
extern void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv,
    double *B, int *ldb, int *info);
extern void zgesv_(int *n, int *nrhs, void *A, int *lda, int *ipiv,
    void *B, int *ldb, int *info);

extern void dgbtrf_(int *m, int *n, int *kl, int *ku, double *AB,
    int *ldab, int *ipiv, int *info);
extern void zgbtrf_(int *m, int *n, int *kl, int *ku, void *AB,
    int *ldab, int *ipiv, int *info);
extern void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
    double *AB, int *ldab, int *ipiv, double *B, int *ldB, int *info);
extern void zgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
    void *AB, int *ldab, int *ipiv, void *B, int *ldB, int *info);
extern void dgbsv_(int *n, int *kl, int *ku, int *nrhs, double *ab,
    int *ldab, int *ipiv, double *b, int *ldb, int *info);
extern void zgbsv_(int *n, int *kl, int *ku, int *nrhs, void *ab,
    int *ldab, int *ipiv, void *b, int *ldb, int *info);

extern void dgttrf_(int *n, double *dl, double *d, double *du,
    double *du2, int *ipiv, int *info);
extern void zgttrf_(int *n, void *dl, void *d, void *du,
    void *du2, int *ipiv, int *info);
extern void dgttrs_(char *trans, int *n, int *nrhs, double *dl, double *d,
    double *du, double *du2, int *ipiv, double *B, int *ldB, int *info);
extern void zgttrs_(char *trans, int *n, int *nrhs, void *dl,
    void *d, void *du, void *du2, int *ipiv, void *B,
    int *ldB, int *info);
extern void dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du,
    double *B, int *ldB, int *info);
extern void zgtsv_(int *n, int *nrhs, void *dl, void *d, void *du,
    void *B, int *ldB, int *info);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);
extern void zpotrs_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    void *B, int *ldb, int *info);
extern void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotri_(char *uplo, int *n, void *A, int *lda, int *info);
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);
extern void zposv_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    void *B, int *ldb, int *info);

extern void dpbtrf_(char *uplo, int *n, int *kd, double *AB, int *ldab,
    int *info);
extern void zpbtrf_(char *uplo, int *n, int *kd, void *AB, int *ldab,
    int *info);
extern void dpbtrs_(char *uplo, int *n, int *kd, int *nrhs, double *AB,
    int *ldab, double *B, int *ldb, int *info);
extern void zpbtrs_(char *uplo, int *n, int *kd, int *nrhs, void *AB,
    int *ldab, void *B, int *ldb, int *info);
extern void dpbsv_(char *uplo, int *n, int *kd, int *nrhs, double *A,
    int *lda, double *B, int *ldb, int *info);
extern void zpbsv_(char *uplo, int *n, int *kd, int *nrhs, void *A,
    int *lda, void *B, int *ldb, int *info);

extern void dpttrf_(int *n, double *d, double *e, int *info);
extern void zpttrf_(int *n, double *d, void *e, int *info);
extern void dpttrs_(int *n, int *nrhs, double *d, double *e, double *B,
    int *ldB, int *info);
extern void zpttrs_(char *uplo, int *n, int *nrhs, double *d, void *e,
    void *B, int *ldB, int *info);
extern void dptsv_(int *n, int *nrhs, double *d, double *e, double *B,
    int *ldB, int *info);
extern void zptsv_(int *n, int *nrhs, double *d, void *e, void *B,
    int *ldB, int *info);

extern void dsytrf_(char *uplo, int *n, double *A, int *lda, int *ipiv,
    double *work, int *lwork, int *info);
extern void zsytrf_(char *uplo, int *n, void *A, int *lda, int *ipiv,
    void *work, int *lwork, int *info);
extern void zhetrf_(char *uplo, int *n, void *A, int *lda, int *ipiv,
    void *work, int *lwork, int *info);
extern void dsytrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, int *info);
extern void zsytrs_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    int *ipiv, void *B, int *ldb, int *info);
extern void zhetrs_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    int *ipiv, void *B, int *ldb, int *info);
extern void dsytri_(char *uplo, int *n, double *A, int *lda, int *ipiv,
    double *work, int *info);
extern void zsytri_(char *uplo, int *n, void *A, int *lda, int *ipiv,
    void *work, int *info);
extern void zhetri_(char *uplo, int *n, void *A, int *lda, int *ipiv,
    void *work, int *info);
extern void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, double *work, int *lwork,
    int *info);
extern void zsysv_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    int *ipiv, void *B, int *ldb, void *work, int *lwork, int *info);
extern void zhesv_(char *uplo, int *n, int *nrhs, void *A, int *lda,
    int *ipiv, void *B, int *ldb, void *work, int *lwork, int *info);

extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
    double  *a, int *lda, double *b, int *ldb, int *info);
extern void ztrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
    void  *a, int *lda, void *b, int *ldb, int *info);
extern void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda,
    int *info);
extern void ztrtri_(char *uplo, char *diag, int *n, void  *a, int *lda,
    int *info);
extern void dtbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd,
    int *nrhs, double *ab, int *ldab, double *b, int *ldb, int *info);
extern void ztbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd,
    int *nrhs, void *ab, int *ldab, void *b, int *ldb, int *info);

extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a,
    int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
extern void zgels_(char *trans, int *m, int *n, int *nrhs, void *a,
    int *lda, void *b, int *ldb, void *work, int *lwork, int *info);
extern void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);
extern void zgeqrf_(int *m, int *n, void *a, int *lda, void *tau,
    void *work, int *lwork, int *info);
extern void dormqr_(char *side, char *trans, int *m, int *n, int *k,
    double *a, int *lda, double *tau, double *c, int *ldc, double *work,
    int *lwork, int *info);
extern void zunmqr_(char *side, char *trans, int *m, int *n, int *k,
    void *a, int *lda, void *tau, void *c, int *ldc,
    void *work, int *lwork, int *info);
extern void dorgqr_(int *m, int *n, int *k, double *A, int *lda,
    double *tau, double *work, int *lwork, int *info);
extern void zungqr_(int *m, int *n, int *k, void *A, int *lda,
    void *tau, void *work, int *lwork, int *info);
extern void dorglq_(int *m, int *n, int *k, double *A, int *lda,
    double *tau, double *work, int *lwork, int *info);
extern void zunglq_(int *m, int *n, int *k, void *A, int *lda,
    void *tau, void *work, int *lwork, int *info);

extern void dgelqf_(int *m, int *n, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);
extern void zgelqf_(int *m, int *n, void *a, int *lda, void *tau,
    void *work, int *lwork, int *info);
extern void dormlq_(char *side, char *trans, int *m, int *n, int *k,
    double *a, int *lda, double *tau, double *c, int *ldc, double *work,
    int *lwork, int *info);
extern void zunmlq_(char *side, char *trans, int *m, int *n, int *k,
    void *a, int *lda, void *tau, void *c, int *ldc,
    void *work, int *lwork, int *info);

extern void dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt,
    double *tau, double *work, int *lwork, int *info);
extern void zgeqp3_(int *m, int *n, void *a, int *lda, int *jpvt,
    void *tau, void *work, int *lwork, double *rwork, int *info);

extern void dsyev_(char *jobz, char *uplo, int *n, double *A, int *lda,
    double *W, double *work, int *lwork, int *info);
extern void zheev_(char *jobz, char *uplo, int *n, void *A, int *lda,
    double *W, void *work, int *lwork, double *rwork, int *info);
extern void dsyevx_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *lda, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldz, double *work, int *lwork,
    int *iwork, int *ifail, int *info);
extern void zheevx_(char *jobz, char *range, char *uplo, int *n,
    void *A, int *lda, double *vl, double *vu, int *il, int *iu,
    double *abstol, int *m, double *W, void *Z, int *ldz, void *work,
    int *lwork, double *rwork, int *iwork, int *ifail, int *info);
extern void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA,
    double *W, double *work, int *lwork, int *iwork, int *liwork,
    int *info);
extern void zheevd_(char *jobz, char *uplo, int *n, void *A, int *ldA,
    double *W, void *work, int *lwork, double *rwork, int *lrwork,
    int *iwork, int *liwork, int *info);
extern void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldZ, int *isuppz, double *work,
    int *lwork, int *iwork, int *liwork, int *info);
extern void zheevr_(char *jobz, char *range, char *uplo, int *n,
    void *A, int *ldA, double *vl, double *vu, int *il, int *iu,
    double *abstol, int *m, double *W, void *Z, int *ldZ, int *isuppz,
    void *work, int *lwork, double *rwork, int *lrwork, int *iwork,
    int *liwork, int *info);

extern void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *A,
    int *lda, double *B, int *ldb, double *W, double *work, int *lwork,
    int *info);
extern void zhegv_(int *itype, char *jobz, char *uplo, int *n, void *A,
    int *lda, void *B, int *ldb, double *W, void *work, int *lwork,
    double *rwork, int *info);

extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A,
    int *ldA, double *S, double *U, int *ldU, double *Vt, int *ldVt,
    double *work, int *lwork, int *info);
extern void dgesdd_(char *jobz, int *m, int *n, double *A, int *ldA,
    double *S, double *U, int *ldU, double *Vt, int *ldVt, double *work,
    int *lwork, int *iwork, int *info);
extern void zgesvd_(char *jobu, char *jobvt, int *m, int *n, void *A,
    int *ldA, double *S, void *U, int *ldU, void *Vt, int *ldVt,
    void *work, int *lwork, double *rwork, int *info);
extern void zgesdd_(char *jobz, int *m, int *n, void *A, int *ldA,
    double *S, void *U, int *ldU, void *Vt, int *ldVt, void *work,
    int *lwork, double *rwork, int *iwork, int *info);

extern void dgees_(char *jobvs, char *sort, void *select, int *n,
    double *A, int *ldA, int *sdim, double *wr, double *wi, double *vs,
    int *ldvs, double *work, int *lwork, int *bwork, int *info);
extern void zgees_(char *jobvs, char *sort, void *select, int *n,
    void *A, int *ldA, int *sdim, void *w, void *vs, int *ldvs,
    void *work, int *lwork, void *rwork, int *bwork, int *info);
extern void dgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg,
    int *n, double *A, int *ldA, double *B, int *ldB, int *sdim,
    double *alphar, double *alphai, double *beta, double *vsl, int *ldvsl,
    double *vsr, int *ldvsr, double *work, int *lwork, int *bwork,
    int *info);
extern void zgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg,
    int *n, void *A, int *ldA, void *B, int *ldB, int *sdim,
    void *alpha, void *beta, void *vsl, int *ldvsl, void *vsr,
    int *ldvsr, void *work, int *lwork, double *rwork, int *bwork,
    int *info);

#endif
