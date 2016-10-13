#include "tempo2.h"

// originally declared in t2fit_stdFitFuncs.h

double t2FitFunc_zero(struct pulsar *psr,int ipsr,double x,int ipos,param_label label,int k);
void t2UpdateFunc_zero(struct pulsar *psr,int ipsr,param_label label,int k,double val,double err);

double t2FitFunc_jump(struct pulsar *psr,int ipsr,double x,int ipos,param_label label,int k);
void t2UpdateFunc_jump(struct pulsar *psr,int ipsr,param_label label,int k,double val,double err);

// defined in t2fit.C, not declared in t2fit.h

void t2fit_fillOneParameterFitInfo(struct pulsar *psr,param_label fit_param,const int k,FitInfo& OUT);
