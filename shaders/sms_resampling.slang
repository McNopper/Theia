// GRIS 2022 Eq.37 / Hong 2025 Eq.10-11, one canonical and N neighboring
// UNMERGED domains. Proxy targets are not PDFs or reciprocal-count estimates.
bool smsPairwiseMis(float target,float source,uint neighbors,bool supported,
    out float canonicalPart,out float neighborWeight) {
    canonicalPart=neighborWeight=0.0;
    if (neighbors==0u || !isfinite(target) || target<0.0) return false;
    if (!supported) source=0.0;
    if (!isfinite(source) || source<0.0) return false;
    float scale=max(target,source);
    if (scale==0.0) return true; // outside the nonzero target support
    float t=target/scale,s=source/scale;
    float denominator=t+float(neighbors)*s;
    canonicalPart=(t/denominator)/float(neighbors);
    neighborWeight=s/denominator;
    return isfinite(canonicalPart) && isfinite(neighborWeight);
}

struct SmsResamplingState {
    float weightSum,selectedTarget;
    uint selected;
};
SmsResamplingState emptySmsResampling() {
    SmsResamplingState state;
    state.weightSum=state.selectedTarget=0.0; state.selected=0xFFFFFFFFu;
    return state;
}
bool smsResamplingAdd(inout SmsResamplingState state,uint candidate,float target,float ucw,float mis,float u) {
    if (!isfinite(target) || !isfinite(ucw) || !isfinite(mis) || !isfinite(u) ||
        target<0.0 || ucw<0.0 || mis<0.0 || u<0.0 || u>=1.0) return false;
    float weight=target*(mis*ucw),total=state.weightSum+weight;
    if (!isfinite(weight) || !isfinite(total)) return false;
    if (weight>0.0 && u*total<weight) {
        state.selected=candidate; state.selectedTarget=target;
    }
    state.weightSum=total;
    return true;
}
bool smsResamplingEstimate(SmsResamplingState state,float3 integrand,out float3 estimate) {
    estimate=0.0.xxx;
    if (!all(isfinite(integrand)) || any(integrand<0.0.xxx) || !isfinite(state.weightSum) || state.weightSum<0.0) return false;
    if (state.weightSum==0.0) return true;
    if (state.selected==0xFFFFFFFFu || !(state.selectedTarget>0.0) || !isfinite(state.selectedTarget)) return false;
    // Divide the color by its target first; do not overflow an intermediate UCW
    // when the final radiometric estimate is representable.
    estimate=(integrand/state.selectedTarget)*state.weightSum;
    return all(isfinite(estimate));
}

// A finite sentinel denotes exact zero support. All positive quantities are
// log2 values; even physical F / UCW outside the float range remain supported.
static const float kSmsLogZero=-3.402823466e38;
bool smsLogNonnegative(float value,out float logarithm) {
    uint bits=asuint(value),magnitude=bits&0x7FFFFFFFu;
    logarithm=kSmsLogZero;
    if (magnitude==0u) return true;
    if ((bits&0x80000000u)!=0u || magnitude>=0x7F800000u) return false;
    uint exponent=bits>>23u,fraction=bits&0x007FFFFFu;
    // Decode subnormals before log2, independent of arithmetic denormal mode.
    logarithm=exponent==0u ? log2(float(fraction))-149.0 :
        float(int(exponent)-127)+log2(asfloat(0x3F800000u|fraction));
    return isfinite(logarithm);
}
float smsLogAdd(float a,float b) {
    if (a==kSmsLogZero) return b;
    if (b==kSmsLogZero) return a;
    return max(a,b)+log2(1.0+exp2(-abs(a-b)));
}
bool smsLogRgb(float3 value,out float3 logarithm) {
    logarithm=kSmsLogZero.xxx;
    return smsLogNonnegative(value.x,logarithm.x) &&
        smsLogNonnegative(value.y,logarithm.y) && smsLogNonnegative(value.z,logarithm.z);
}
float smsLogScale(float value,float scale) { return value==kSmsLogZero ? value : value+scale; }
float3 smsLogScaleRgb(float3 value,float scale) {
    return float3(smsLogScale(value.x,scale),smsLogScale(value.y,scale),smsLogScale(value.z,scale));
}
float smsLogTarget(float3 logarithm) {
    return smsLogAdd(smsLogAdd(smsLogScale(logarithm.x,log2(.2126)),
        smsLogScale(logarithm.y,log2(.7152))),smsLogScale(logarithm.z,log2(.0722)));
}
bool smsCacheLocalTarget(bool enabled,float3 weighted,float lightPdf,inout float cached) {
    if (!enabled) return true; // unused cache must not fail a local-only frame
    float3 logarithm; float pdf;
    if (!smsLogRgb(weighted,logarithm) || !smsLogNonnegative(lightPdf,pdf) || pdf==kSmsLogZero) return false;
    cached=smsLogScale(smsLogTarget(logarithm),pdf);
    return isfinite(cached);
}
bool smsLogUnbiasedWeight(float reciprocal,float conditionalMass,float lightPdf,out float logWeight) {
    logWeight=kSmsLogZero;
    float count,mass,pdf;
    if (!smsLogNonnegative(reciprocal,count) || !smsLogNonnegative(conditionalMass,mass) ||
        !smsLogNonnegative(lightPdf,pdf) || count==kSmsLogZero || mass==kSmsLogZero || pdf==kSmsLogZero) return false;
    logWeight=count-mass-pdf;
    return isfinite(logWeight);
}
bool smsLogContribution(float3 base,float3 response,float geometric,float3 emission,
    float pdf,float3 prefix,out float3 value) {
    value=kSmsLogZero.xxx;
    float3 b,r,e,p; float g,q;
    if (!smsLogRgb(base,b) || !smsLogRgb(response,r) || !smsLogRgb(emission,e) ||
        !smsLogRgb(prefix,p) || !smsLogNonnegative(geometric,g) ||
        !smsLogNonnegative(pdf,q) || q==kSmsLogZero) return false;
    for (uint i=0u;i<3u;++i) {
        if (b[i]==kSmsLogZero || r[i]==kSmsLogZero || e[i]==kSmsLogZero || p[i]==kSmsLogZero || g==kSmsLogZero) continue;
        value[i]=b[i]+r[i]+g+e[i]-q+p[i];
    }
    return all(isfinite(value));
}
bool smsLogPairwiseMis(float target,float source,uint neighbors,bool supported,
    out float canonicalPart,out float neighborWeight) {
    canonicalPart=neighborWeight=kSmsLogZero;
    if (neighbors==0u || !isfinite(target) || !isfinite(source)) return false;
    if (!supported) source=kSmsLogZero;
    if (target==kSmsLogZero && source==kSmsLogZero) return true;
    float n=log2(float(neighbors)),denominator=smsLogAdd(target,smsLogScale(source,n));
    if (target!=kSmsLogZero) canonicalPart=target-denominator-n;
    if (source!=kSmsLogZero) neighborWeight=source-denominator;
    return isfinite(canonicalPart) && isfinite(neighborWeight);
}
struct SmsLogResamplingState {
    float weightSum,selectedTarget;
    uint selected;
};
SmsLogResamplingState emptySmsLogResampling() {
    SmsLogResamplingState state;
    state.weightSum=state.selectedTarget=kSmsLogZero; state.selected=0xFFFFFFFFu;
    return state;
}
bool smsLogResamplingAdd(inout SmsLogResamplingState state,uint candidate,
    float target,float ucw,float mis,float u) {
    if (!isfinite(target) || !isfinite(ucw) || !isfinite(mis) || !isfinite(u) || u<0.0 || u>=1.0) return false;
    if (target==kSmsLogZero || ucw==kSmsLogZero || mis==kSmsLogZero) return true;
    float weight=target+ucw+mis,total=smsLogAdd(state.weightSum,weight);
    if (!isfinite(weight) || !isfinite(total)) return false;
    // Select the first positive candidate unconditionally. Compare a bounded
    // probability, not log(u)+total: that sum loses a near-one draw at large
    // log magnitudes and can leave a positive reservoir without any selection.
    if (state.weightSum==kSmsLogZero || u==0.0 || u<exp2(weight-total)) {
        state.selected=candidate; state.selectedTarget=target;
    }
    state.weightSum=total;
    return true;
}
bool smsLogResamplingEstimate(SmsLogResamplingState state,float3 integrand,out float3 estimate) {
    estimate=0.0.xxx;
    if (!all(isfinite(integrand)) || !isfinite(state.weightSum)) return false;
    if (state.weightSum==kSmsLogZero) return true;
    if (state.selected==0xFFFFFFFFu || state.selectedTarget==kSmsLogZero || !isfinite(state.selectedTarget)) return false;
    for (uint i=0u;i<3u;++i)
        if (integrand[i]!=kSmsLogZero) estimate[i]=exp2(integrand[i]-state.selectedTarget+state.weightSum);
    return all(isfinite(estimate)); // only the final representability is required
}
