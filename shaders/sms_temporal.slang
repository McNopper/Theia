// Hong et al. 2025, section 5: temporal reservoir merge. Dispatched once per
// ready batch like validation; internally gated on the same ready status so the
// merge runs exactly once per logical frame, after every slot is Done and
// before resolve. Candidate domains: the current pixel's fresh SMS candidate
// and the reprojected previous-frame reservoir. Both directional shifts are
// bounded (5-iteration) manifold walks with inverse-root checks; failed shifts
// define zero MIS support. This pass never reads or writes the HDR target.
import common;
import math;
import rng;
import bsdf;
import medium_walk;
import surface;
import sms;
import sms_transport;
import sms_work;
import sms_scene;
import sms_context;
import sms_shift;
import sms_resampling;
[[vk::push_constant]] SmsQueuePush pc;
#define SMS_HELPERS_ONLY 1
#define SMS_SCENE TheiaSmsScene
#define SMS_HIT TheiaSmsHit
#define SMS_TEXTURE_SAMPLER TheiaSmsTextureSampler
#define SMS_MATERIALS t_smsMaterials
#define SMS_EMITTERS t_smsEmitters
#define SMS_EMITTER_CDF t_smsEmitterCdf
#define SMS_CONTRIBUTION_ZERO kSmsLogZero.xxx
#define SMS_FINISH_CONTRIBUTION(base,response,geometric,Le,pdf,prefix,value) \
    return smsLogContribution(base,response,geometric,Le,pdf,prefix,value);
#include "sms_compute.slang"
#include "sms_reuse.slang"

static const float kSmsTemporalMassCap=20.0;

float2 smsTemporalJitter(SmsWorkRecord work,uint slot) {
    float4 draw=smsTrialSample(work.words[7].zw,uint2(slot,0u),uint2(0xFFFFFFFAu,0));
    return draw.xy-0.5.xx;
}

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    uint slot;
    if (!smsQueueValidationSlot(pc,group,lane,slot)) return;
    smsDiagnostic(22u);
    GiSmsContext context=t_smsProducer[0];
    SmsReservoir merged=smsEmptyReservoir();
    context.temporalReservoir[slot]=merged;
    SmsWorkRecord work=pc.work[slot];
    uint width=work.words[15].z;
    if (smsPhase(work)!=kSmsDone || width==0u) return;
    float4 prefix=context.prefixes[slot];
    bool centralCandidate;
    float centralUcwLog;
    if (!smsLocalCandidate(work,prefix.w,centralCandidate,centralUcwLog)) {
        smsDiagnostic(24u);
        return;
    }
    if (!centralCandidate) return;
    smsDiagnostic(23u);
    uint2 pixel=uint2(slot%width,slot/width);
    uint height=work.words[15].w;
    if (slot/width!=pixel.y || pixel.x>=width) return;
    SmsManifoldVertex centralRoot[2];
    uint centralCount;
    smsLoadRoot(work,centralRoot,centralCount);
    float centralMass=1.0;
    float centralUcw=exp2(centralUcwLog);
    smsDiagnosticMaxFloat(33u,centralUcw);
    float3 centralEstimate=asfloat(work.words[11].xyz);
    float centralLuminance=smsLuminance(centralEstimate);
    smsDiagnosticMaxFloat(34u,centralLuminance);
    if (!(centralLuminance>0.0) || !isfinite(centralLuminance)) return;
    // Temporal domain: reprojected previous-frame reservoir. Clamp the fetch to
    // the bound image extent (the 1x1 dummy must never be read out of bounds).
    uint motionWidth,motionHeight;
    t_smsMotionVectors.GetDimensions(motionWidth,motionHeight);
    uint2 fetchPixel=min(pixel,uint2(motionWidth-1u,motionHeight-1u));
    float2 motion=t_smsMotionVectors[fetchPixel].xy;
    if (!all(isfinite(motion))) return;
    int2 previousPixel=int2(round(float2(pixel)+motion+smsTemporalJitter(work,slot)));
    bool temporalDomain=all(previousPixel>=int2(0,0)) &&
        previousPixel.x<int(width) && previousPixel.y<int(height);
    SmsReservoir previous=context.finalReservoir[temporalDomain?uint(previousPixel.y*width+previousPixel.x):slot];
    float previousMass=smsReservoirValid(previous)?min(asfloat(previous.mass),kSmsTemporalMassCap):0.0;
    smsDiagnosticMaxFloat(26u,previousMass);
    TheiaSmsScene scene;
    bool temporalSupported=false;
    float3 temporalEstimateLinear=0.0.xxx;
    SmsManifoldVertex shiftedTemporal[2]={(SmsManifoldVertex)0,(SmsManifoldVertex)0};
    uint temporalCount=0u;
    float temporalLuminance=0.0;
    if (temporalDomain && previousMass>0.0) {
        SmsManifoldVertex previousRoot[2];
        uint previousCount;
        smsLoadRoot(previous.record,previousRoot,previousCount);
        uint shiftFailures;
        if (smsShiftInverseChecked(scene,smsReceiverPosition(previous.record),previous.record.words[13].w,
            smsReceiverPosition(work),work.words[13].w,smsLightPosition(previous.record),
            previousRoot,previousCount,shiftedTemporal,shiftFailures)) {
            float3 value;
            // A shifted path may fail admissibility or support; that is zero
            // candidate support, not a numeric fault (Hong et al. 2025: failed
            // shifts contribute nothing).
            if (smsShiftIntegrand(scene,work,prefix.xyz,previous.record,shiftedTemporal,previousCount,value)) {
                temporalEstimateLinear=smsReservoirLinearFromLog(value);
                temporalLuminance=smsLuminance(temporalEstimateLinear);
                if (temporalLuminance>0.0 && isfinite(temporalLuminance) && asfloat(previous.weight)>0.0) {
                    temporalSupported=true;
                    temporalCount=previousCount;
                }
            }
        }
    }
    // Backward shift supports the central candidate's MIS weight.
    float centralSupportedLuminance=0.0;
    bool centralSupported=false;
    if (temporalDomain && previousMass>0.0) {
        SmsManifoldVertex shiftedCentral[2]={(SmsManifoldVertex)0,(SmsManifoldVertex)0};
        uint shiftFailures;
        if (smsShiftInverseChecked(scene,smsReceiverPosition(work),work.words[13].w,
            smsReceiverPosition(previous.record),previous.record.words[13].w,smsLightPosition(work),
            centralRoot,centralCount,shiftedCentral,shiftFailures)) {
            float3 value;
            if (smsShiftIntegrand(scene,previous.record,previous.prefix.xyz,work,shiftedCentral,centralCount,value)) {
                centralSupportedLuminance=smsLogLuminanceToLinear(value);
                if (centralSupportedLuminance>0.0 && isfinite(centralSupportedLuminance)) centralSupported=true;
            }
        }
    }
    // Defensive two-domain MIS masses (Hong et al. 2025, section 5.1).
    float previousLuminance=smsLuminance(previous.estimate);
    float temporalMis=0.0;
    float centralMis=1.0;
    if (temporalSupported && previousLuminance>0.0) {
        float temporalTerm=previousMass*previousLuminance;
        float centralTerm=centralMass*temporalLuminance;
        temporalMis=temporalTerm>0.0 ? temporalTerm/(temporalTerm+centralTerm) : 0.0;
        if (centralSupported && centralSupportedLuminance>0.0) {
            float centralOwnTerm=centralMass*centralLuminance;
            float backwardTerm=previousMass*centralSupportedLuminance;
            centralMis=centralOwnTerm>0.0 ? centralOwnTerm/(centralOwnTerm+backwardTerm) : 0.0;
        } else centralMis=1.0;
    }
    float temporalUcw=asfloat(previous.weight);
    float temporalWeight=temporalMis*temporalLuminance*temporalUcw;
    float centralWeight=centralMis*centralLuminance*centralUcw;
    smsDiagnosticMaxFloat(35u,temporalWeight);
    smsDiagnosticMaxFloat(36u,centralWeight);
    float weightSum=centralWeight+temporalWeight;
    if (!(weightSum>0.0) || !isfinite(weightSum)) return;
    float4 draw=smsTrialSample(work.words[7].zw,uint2(slot,1u),uint2(0xFFFFFFF9u,0));
    bool selectTemporal=draw.x*weightSum<temporalWeight;
    if (selectTemporal) smsDiagnostic(37u);
    // Reference finalizeGRIS: the reservoir carries the GRIS estimate weight
    // (raw sum over the selected candidate's target luminance). Without this
    // division the scale compounds across frames and the estimate explodes.
    float selectedLuminance=selectTemporal?temporalLuminance:centralLuminance;
    if (!(selectedLuminance>0.0) || !isfinite(selectedLuminance)) return;
    smsDiagnosticMaxFloat(38u,selectedLuminance);
    merged.record=work;
    merged.prefix=prefix;
    if (selectTemporal) {
        merged.record.words[5]=previous.record.words[5];
        merged.record.words[6]=previous.record.words[6];
        merged.record.words[7].xy=previous.record.words[7].xy;
        smsStoreRoot(merged.record,shiftedTemporal,temporalCount);
        merged.estimate=temporalEstimateLinear;
        merged.count=temporalCount;
    } else {
        merged.estimate=centralEstimate;
        merged.count=centralCount;
    }
    merged.weight=asuint(weightSum/selectedLuminance);
    merged.mass=asuint(centralMass+(temporalSupported?previousMass:0.0));
    smsDiagnosticMaxFloat(25u,asfloat(merged.weight));
    smsDiagnosticMaxFloat(26u,previousMass);
    context.temporalReservoir[slot]=merged;
}