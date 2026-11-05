// Included by the native resolve after the common radiometric helpers. Inputs
// remain immutable/unmerged; only the final HDR pixel receives resampled energy.
static const uint kSmsSpatialNeighbors=3u;
float smsReuseTarget(float3 value) { return smsLogTarget(value); }

bool smsShiftIntegrand(inout TheiaSmsScene scene,SmsWorkRecord receiver,float3 prefix,
    SmsWorkRecord emitter,SmsManifoldVertex root[2],uint count,out float3 value) {
    SmsWorkRecord work=receiver;
    work.words[5]=emitter.words[5]; work.words[5].w=asuint(1.0); // physical F, not F/p_light
    work.words[6]=emitter.words[6]; work.words[7].xy=emitter.words[7].xy;
    work.words[9].xyz=asuint(prefix);
    return contribution(work,root,count,value);
}

bool smsSpatialEstimateRecords(uint slot,uint width,uint height,SmsWorkRecord canonical,out float3 estimate) {
    estimate=0.0.xxx;
    if (smsPhase(canonical)==kSmsAbsent) return true;
    GiSmsContext context=t_smsProducer[0];
    float4 prefix=context.prefixes[slot];
    bool localCandidate; float localUcw;
    if (!smsLocalCandidate(canonical,prefix.w,localCandidate,localUcw)) return false;
    if (localCandidate) smsDiagnostic(kSmsDiagCanonicalCandidates);
    SmsManifoldVertex original[2]; uint originalCount=0u;
    original[0]=original[1]=(SmsManifoldVertex)0;
    if (localCandidate) smsLoadRoot(canonical,original,originalCount);
    float3 values[4]; float targets[4],ucws[4],weights[4];
    for (uint i=0u;i<4u;++i) { values[i]=kSmsLogZero.xxx; targets[i]=ucws[i]=weights[i]=kSmsLogZero; }
    if (localCandidate) {
        if (!smsLogRgb(asfloat(canonical.words[11].xyz),values[0])) return false;
        values[0]=smsLogScaleRgb(values[0],-localUcw);
        targets[0]=prefix.w; ucws[0]=localUcw;
    }
    uint2 pixel=uint2(slot%width,slot/width);
    uint selectedSlots[3]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    TheiaSmsScene scene;
    for (uint i=0u;i<kSmsSpatialNeighbors;++i) {
        float4 draw=smsTrialSample(canonical.words[7].zw,uint2(i,0),uint2(0xFFFFFFFCu,0));
        int2 delta=int2(floor(draw.xy*11.0))-5;
        if (all(delta==int2(0,0))) delta=i==0u?int2(1,0):i==1u?int2(0,1):int2(-1,0);
        int2 neighbor=int2(pixel)+delta;
        bool domain=all(neighbor>=int2(0,0)) && all(neighbor<int2(width,height));
        uint neighborSlot=0u;
        if (domain) neighborSlot=uint(neighbor.y)*width+uint(neighbor.x);
        for (uint previous=0u;previous<i;++previous)
            if (domain && selectedSlots[previous]==neighborSlot) domain=false;
        selectedSlots[i]=domain?neighborSlot:0xFFFFFFFFu;
        SmsWorkRecord source={}; float4 sourcePrefix=0.0.xxxx;
        bool sourceCandidate=false; float sourceUcw=0.0;
        if (domain) {
            source=pc.work[neighborSlot]; sourcePrefix=context.prefixes[neighborSlot];
            if (any(source.words[15].zw!=uint2(width,height)) ||
                !smsLocalCandidate(source,sourcePrefix.w,sourceCandidate,sourceUcw)) return false;
            domain=smsPhase(source)!=kSmsAbsent;
            if (domain) smsDiagnostic(kSmsDiagNeighborDomains);
        }
        float proxyAtCanonical=kSmsLogZero;
        bool canonicalSupport=false;
        if (localCandidate && domain) {
            smsDiagnostic(kSmsDiagCanonicalAttempts);
            SmsManifoldVertex shifted[2];
            uint shiftFailures;
            canonicalSupport=smsShiftInverseChecked(scene,smsReceiverPosition(canonical),canonical.words[13].w,
                smsReceiverPosition(source),source.words[13].w,smsLightPosition(canonical),original,originalCount,shifted,shiftFailures);
            if ((shiftFailures&kSmsShiftSupportFailure)!=0u) smsDiagnostic(kSmsDiagSupportFailures);
            if ((shiftFailures&kSmsShiftTopologyFailure)!=0u) smsDiagnostic(kSmsDiagTopologyFailures);
            if ((shiftFailures&kSmsShiftSolverFailure)!=0u) smsDiagnostic(kSmsDiagSolverFailures);
            if ((shiftFailures&kSmsShiftVisibilityFailure)!=0u) smsDiagnostic(kSmsDiagVisibilityFailures);
            if (canonicalSupport) {
                smsDiagnostic(kSmsDiagCanonicalAccepts);
                float3 sourceValue;
                if (!smsShiftIntegrand(scene,source,sourcePrefix.xyz,canonical,shifted,originalCount,sourceValue)) return false;
                proxyAtCanonical=smsReuseTarget(sourceValue);
            }
        }
        if (localCandidate) {
            float part,unused;
            if (!smsLogPairwiseMis(prefix.w,proxyAtCanonical,kSmsSpatialNeighbors,canonicalSupport,part,unused)) return false;
            weights[0]=smsLogAdd(weights[0],part);
        }
        if (!domain || !sourceCandidate) continue;
        smsDiagnostic(kSmsDiagNeighborCandidates);
        smsDiagnostic(kSmsDiagNeighborAttempts);
        SmsManifoldVertex sourceRoot[2],shifted[2]; uint count;
        smsLoadRoot(source,sourceRoot,count);
        uint shiftFailures;
        if (!smsShiftInverseChecked(scene,smsReceiverPosition(source),source.words[13].w,
            smsReceiverPosition(canonical),canonical.words[13].w,smsLightPosition(source),sourceRoot,count,shifted,shiftFailures)) {
            if ((shiftFailures&kSmsShiftSupportFailure)!=0u) smsDiagnostic(kSmsDiagSupportFailures);
            if ((shiftFailures&kSmsShiftTopologyFailure)!=0u) smsDiagnostic(kSmsDiagTopologyFailures);
            if ((shiftFailures&kSmsShiftSolverFailure)!=0u) smsDiagnostic(kSmsDiagSolverFailures);
            if ((shiftFailures&kSmsShiftVisibilityFailure)!=0u) smsDiagnostic(kSmsDiagVisibilityFailures);
            continue;
        }
        smsDiagnostic(kSmsDiagNeighborAccepts);
        float3 targetValue;
        if (!smsShiftIntegrand(scene,canonical,prefix.xyz,source,shifted,count,targetValue)) return false;
        float target=smsReuseTarget(targetValue),unused,mis;
        if (!smsLogPairwiseMis(target,sourcePrefix.w,kSmsSpatialNeighbors,true,unused,mis)) return false;
        values[i+1u]=targetValue; targets[i+1u]=target; ucws[i+1u]=sourceUcw; weights[i+1u]=mis;
    }
    SmsLogResamplingState reservoir=emptySmsLogResampling();
    float4 selection=smsTrialSample(canonical.words[7].zw,uint2(0,0),uint2(0xFFFFFFFBu,0));
    for (uint i=0u;i<4u;++i)
        if (!smsLogResamplingAdd(reservoir,i,targets[i],ucws[i],weights[i],selection[i])) return false;
    if (reservoir.weightSum==kSmsLogZero) return true;
    if (reservoir.selected>=4u) return false;
    if (reservoir.selected>0u) smsDiagnostic(kSmsDiagSelectedNeighbors);
    return smsLogResamplingEstimate(reservoir,values[reservoir.selected],estimate);
}

// Defensive pairwise MIS masses (Hong et al. 2025 section 5.1; the reference's
// ReSTIRCommon formulas). All arguments are linear-domain positive floats.
float smsSpatialNeighborMis(float massSum,float neighborMass,float neighborLuminance,
    float centralMass,float shiftedLuminance) {
    float denominator=(massSum-centralMass)*neighborLuminance;
    if (!(denominator>0.0) || !(massSum>0.0)) return 0.0;
    return (neighborMass/massSum)*denominator/(denominator+centralMass*shiftedLuminance);
}
// pairwiseMIS_defensiveCanonical(cSum, c0=M_n, pi=lum(Tp), c1=M_c, pc): the
// prefactor mass is the NEIGHBOR'S (c0), not the central's — with the central
// mass there the pairwise weights only sum to one at M_c == M_n (measured
// 0.875 at {1,3}). A failed backward shift (pi = 0) collapses the fraction to
// M_n/cSum: the neighbor's probability mass absorbs into the canonical weight.
float smsSpatialCanonicalMis(float massSum,float neighborMass,float centralMass,
    float backwardLuminance,float centralLuminance) {
    float weight=centralMass*centralLuminance;
    float denominator=(massSum-centralMass)*backwardLuminance;
    if (!(weight>0.0) || !(massSum>0.0)) return 0.0;
    return (neighborMass/massSum)*weight/(denominator+weight);
}
bool smsSpatialEstimateReservoirs(uint slot,uint width,uint height,SmsWorkRecord canonical,
    out float3 estimate) {
    estimate=0.0.xxx;
    if (smsPhase(canonical)==kSmsAbsent) return true;
    GiSmsContext context=t_smsProducer[0];
    SmsReservoir central=context.temporalReservoir[slot];
    if (!smsReservoirValid(central)) return true;
    smsDiagnostic(kSmsDiagCanonicalCandidates);
    float4 prefix=context.prefixes[slot];
    float centralMass=asfloat(central.mass);
    float centralLuminance=smsLuminance(central.estimate);
    float3 values[4];
    float targets[4],ucws[4],weights[4];
    for (uint i=0u;i<4u;++i) { values[i]=kSmsLogZero.xxx; targets[i]=ucws[i]=weights[i]=kSmsLogZero; }
    if (!smsLogRgb(central.estimate,values[0])) { smsDiagnostic(29u); return true; }
    // GRIS targets are the candidates' full value luminance in this domain
    // (Hong et al. 2025 section 5.1: p_hat = f). values[0] is the log-domain
    // candidate value; the prefix proxy is not a consistent target against
    // reservoir-scale candidate weights.
    targets[0]=smsLogTarget(values[0]);
    ucws[0]=asfloat(central.weight)>0.0?log2(asfloat(central.weight)):kSmsLogZero;
    weights[0]=kSmsLogZero; // accumulated after the loop from the defensive masses
    uint2 pixel=uint2(slot%width,slot/width);
    uint selectedSlots[3]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    // Reference two-pass structure: cSum = centralM + all valid neighbor M
    // BEFORE any MIS weight is formed (SpatialResampling.cs.slang).
    SmsReservoir sources[3]={smsEmptyReservoir(),smsEmptyReservoir(),smsEmptyReservoir()};
    bool candidates[3]={false,false,false};
    float massSum=centralMass;
    for (uint i=0u;i<kSmsSpatialNeighbors;++i) {
        float4 draw=smsTrialSample(central.record.words[7].zw,uint2(i,0),uint2(0xFFFFFFFCu,0));
        int2 delta=int2(floor(draw.xy*11.0))-5;
        if (all(delta==int2(0,0))) delta=i==0u?int2(1,0):i==1u?int2(0,1):int2(-1,0);
        int2 neighbor=int2(pixel)+delta;
        bool domain=all(neighbor>=int2(0,0)) && all(neighbor<int2(width,height));
        uint neighborSlot=0u;
        if (domain) neighborSlot=uint(neighbor.y)*width+uint(neighbor.x);
        for (uint previous=0u;previous<i;++previous)
            if (domain && selectedSlots[previous]==neighborSlot) domain=false;
        selectedSlots[i]=domain?neighborSlot:0xFFFFFFFFu;
        if (domain) {
            SmsReservoir source=context.temporalReservoir[neighborSlot];
            bool valid=smsReservoirValid(source);
            // Dimension guard applies only to VALID neighbor records (empty
            // entries carry no frame dimensions by construction).
            if (valid && any(source.record.words[15].zw!=uint2(width,height))) { smsDiagnostic(51u); valid=false; }
            sources[i]=source;
            candidates[i]=valid;
            if (valid) {
                smsDiagnostic(kSmsDiagNeighborDomains);
                massSum+=asfloat(source.mass);
            }
        }
    }
    TheiaSmsScene scene;
    SmsManifoldVertex centralRoot[2];
    uint centralCount;
    smsLoadRoot(central.record,centralRoot,centralCount);
    float canonicalMis=centralMass/massSum;
    float misSum=0.0;
    for (uint i=0u;i<kSmsSpatialNeighbors;++i) {
        SmsReservoir source=sources[i];
        bool sourceCandidate=candidates[i];
        float backwardLuminance=0.0;
        bool canonicalSupport=false;
        if (sourceCandidate) {
            smsDiagnostic(kSmsDiagCanonicalAttempts);
            SmsManifoldVertex shifted[2];
            uint shiftFailures;
            canonicalSupport=smsShiftInverseChecked(scene,smsReceiverPosition(central.record),central.record.words[13].w,
                smsReceiverPosition(source.record),source.record.words[13].w,smsLightPosition(central.record),
                centralRoot,centralCount,shifted,shiftFailures);
            if ((shiftFailures&kSmsShiftSupportFailure)!=0u) smsDiagnostic(kSmsDiagSupportFailures);
            if ((shiftFailures&kSmsShiftTopologyFailure)!=0u) smsDiagnostic(kSmsDiagTopologyFailures);
            if ((shiftFailures&kSmsShiftSolverFailure)!=0u) smsDiagnostic(kSmsDiagSolverFailures);
            if ((shiftFailures&kSmsShiftVisibilityFailure)!=0u) smsDiagnostic(kSmsDiagVisibilityFailures);
            if (canonicalSupport) {
                smsDiagnostic(kSmsDiagCanonicalAccepts);
                float3 value;
                // Failed shifted contributions are zero support, not faults.
                // The integrand is log-domain: recover the linear luminance
                // (smsLuminance of raw log values is negative for values < 1
                // and voided every backward shift - the partition leak).
                if (smsShiftIntegrand(scene,source.record,source.prefix.xyz,central.record,shifted,centralCount,value)) {
                    backwardLuminance=smsLogLuminanceToLinear(value);
                    canonicalSupport=backwardLuminance>0.0 && isfinite(backwardLuminance);
                }
            }
        }
        float neighborMass=asfloat(source.mass);
        // Defensive canonical: added for EVERY valid neighbor, failed shifts
        // included — pi = 0 collapses the fraction to M_n/cSum (the defensive
        // absorption). The old `if (canonicalSupport)` guard skipped failed
        // shifts and leaked their probability mass out of the estimator.
        canonicalMis+=smsSpatialCanonicalMis(massSum,neighborMass,centralMass,
            backwardLuminance,centralLuminance);
        if (!sourceCandidate) continue;
        smsDiagnostic(kSmsDiagNeighborCandidates);
        smsDiagnostic(kSmsDiagNeighborAttempts);
        SmsManifoldVertex sourceRoot[2],shifted[2];
        uint count;
        smsLoadRoot(source.record,sourceRoot,count);
        uint shiftFailures;
        if (!smsShiftInverseChecked(scene,smsReceiverPosition(source.record),source.record.words[13].w,
            smsReceiverPosition(central.record),central.record.words[13].w,smsLightPosition(source.record),
            sourceRoot,count,shifted,shiftFailures)) {
            if ((shiftFailures&kSmsShiftSupportFailure)!=0u) smsDiagnostic(kSmsDiagSupportFailures);
            if ((shiftFailures&kSmsShiftTopologyFailure)!=0u) smsDiagnostic(kSmsDiagTopologyFailures);
            if ((shiftFailures&kSmsShiftSolverFailure)!=0u) smsDiagnostic(kSmsDiagSolverFailures);
            if ((shiftFailures&kSmsShiftVisibilityFailure)!=0u) smsDiagnostic(kSmsDiagVisibilityFailures);
            continue;
        }
        smsDiagnostic(kSmsDiagNeighborAccepts);
        float3 targetValue;
        if (!smsShiftIntegrand(scene,central.record,prefix.xyz,source.record,shifted,count,targetValue)) continue;
        float neighborLuminance=smsLuminance(source.estimate);
        float shiftedLuminance=smsLogLuminanceToLinear(targetValue);
        float mis=smsSpatialNeighborMis(massSum,neighborMass,neighborLuminance,centralMass,shiftedLuminance);
        misSum+=mis>0.0?mis:0.0;
        values[i+1u]=targetValue; targets[i+1u]=smsReuseTarget(targetValue);
        ucws[i+1u]=asfloat(source.weight)>0.0?log2(asfloat(source.weight)):kSmsLogZero;
        smsDiagnosticMaxFloat(28u,asfloat(source.weight));
        weights[i+1u]=mis>0.0?log2(mis):kSmsLogZero;
    }
    weights[0]=canonicalMis>0.0?log2(canonicalMis):kSmsLogZero;
    // Partition-of-unity observation: the pairwise weights must sum to one for
    // unbiasedness under value-preserving shifts.
    smsDiagnosticMaxFloat(57u,abs(canonicalMis+misSum-1.0));
    {
        float partition=canonicalMis+misSum;
        if (isfinite(partition) && partition>=0.0) {
            InterlockedAdd(t_smsDiagnostics[60u],uint(min(partition,16.0)*4096.0));
            InterlockedAdd(t_smsDiagnostics[61u],4096u);
        }
    }
    SmsLogResamplingState reservoir=emptySmsLogResampling();
    float4 selection=smsTrialSample(central.record.words[7].zw,uint2(0,0),uint2(0xFFFFFFFBu,0));
    for (uint i=0u;i<4u;++i)
        if (!smsLogResamplingAdd(reservoir,i,targets[i],ucws[i],weights[i],selection[i])) { smsDiagnostic(30u); return true; }
    if (reservoir.weightSum==kSmsLogZero) return true;
    if (reservoir.selected>=4u) { smsDiagnostic(31u); return true; }
    if (reservoir.selected>0u) smsDiagnostic(kSmsDiagSelectedNeighbors);
    smsDiagnosticMaxFloat(27u,asfloat(central.weight));
    if (!smsLogResamplingEstimate(reservoir,values[reservoir.selected],estimate)) {
        // Representability routing: a candidate whose combined GRIS estimate
        // exceeds the float range is dominated by a near-caustic firefly the
        // local estimator also samples. Render the pixel with its local
        // estimate instead of failing the logical frame.
        smsDiagnostic(32u);
        estimate=asfloat(canonical.words[11].xyz);
        return true;
    }
    return true;
}

// Dispatch. When the temporal dispatch is armed (context flag bit 2), the
// reference's population semantics apply: a pixel whose temporal merge produced
// no valid reservoir had no candidate this frame - it shades nothing (its
// energy reaches the image only through other pixels' reservoir imports;
// SpatialResampling.cs.slang line 84 copies the M=0 reservoir and returns). A
// records fallback here would double-count: candidate pixels already carry
// their full local estimate through the reservoir path while imported records
// re-add the same energy at their neighbors. The records path remains only for
// the temporal-disarmed configuration (the pre-temporal reuse behavior).
bool smsSpatialEstimate(uint slot,uint width,uint height,SmsWorkRecord canonical,
    out float3 estimate) {
    estimate=0.0.xxx;
    if (smsPhase(canonical)==kSmsAbsent) return true;
    GiSmsContext context=t_smsProducer[0];
    if ((context.flags&4u)==0u) {
        if (!smsSpatialEstimateRecords(slot,width,height,canonical,estimate)) smsDiagnostic(50u);
        return true; // records-path faults are observation-only during bring-up
    }
    SmsReservoir central=context.temporalReservoir[slot];
    if (!smsReservoirValid(central)) return true;
    if (!smsSpatialEstimateReservoirs(slot,width,height,canonical,estimate)) smsDiagnostic(52u);
    return true; // reservoir-path faults are observation-only during bring-up
}
