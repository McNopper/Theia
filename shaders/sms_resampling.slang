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
