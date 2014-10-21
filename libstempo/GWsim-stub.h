#include "tempo2.h"

#ifdef HAVE_GWSIM_H

#define HAVE_GWSIM 1
#include "GWsim.h"

#else

#define HAVE_GWSIM 0

typedef struct gwSrc {
    long double theta_g;
    long double phi_g;
    long double omega_g;
    long double phi_polar_g;
    long double phase_g;
    long double aplus_g;
    long double aplus_im_g; 
    long double across_g;
    long double across_im_g;
    long double phi_bin;
    long double theta_bin; 
    // long double chirp_mass;
    long double inc_bin;
    long double dist_bin;
    long double h[3][3]; 
    long double h_im[3][3];
    long double kg[3]; 
} gwSrc;

void setupGW(gwSrc *gw);
void GWbackground(gwSrc *gw,int numberGW,long *idum,long double flo,long double fhi,double gwAmp,double alpha,int loglin);
long double calculateResidualGW(long double *kp,gwSrc *gw,long double time,long double dist);
void setupPulsar_GWsim(long double ra_p,long double dec_p,long double *kp);

/* Define the dummy anisotropic functions here */
#warning "Anisotropic GWsim routines not available"

void GWdipolebackground(gwSrc *gw,int numberGW,long *idum,long double flo,long double fhi, double gwAmp,double alpha,int loglin, double *dipoleamps) {
    return;
}

#endif /* HAVE_GWSIM_H */
