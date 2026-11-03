// Local, unmerged SMS estimate. Spatial MIS will consume the completed local
// work records, never feed a recursively merged result back into this source.
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
[[vk::binding(1,0)]] RWTexture2D<float4> t_smsHdr;
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

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 group : SV_GroupID,uint lane : SV_GroupIndex) {
    uint slot;
    if (!smsQueueValidationSlot(pc,group,lane,slot)) return;
    SmsWorkRecord work=pc.work[slot];
    uint width,height;
    t_smsHdr.GetDimensions(width,height);
    if (width==0u || height==0u) {
        smsQueueCompletionError(pc,kSmsQueueInvalidWork); return;
    }
    if (pc.capacity/width!=height || pc.capacity%width!=0u ||
        work.words[14].y!=1u || any(work.words[15].zw!=uint2(width,height))) {
        smsQueueCompletionError(pc,kSmsQueueInvalidWork); return;
    }
    if (smsPhase(work)==kSmsAbsent) return;
    uint2 pixel=uint2(slot%width,slot/width);
    float4 previous=t_smsHdr[pixel];
    float3 estimate=asfloat(work.words[11].xyz);
    if ((t_smsProducer[0].flags&1u)!=0u && !smsSpatialEstimate(slot,width,height,work,estimate)) {
        smsQueueCompletionError(pc,kSmsQueueNumericError); return;
    }
    float3 value=previous.xyz+estimate*asfloat(work.words[14].x);
    if (!all(isfinite(value))) { smsQueueCompletionError(pc,kSmsQueueNumericError); return; }
    smsDiagnosticMaxFloat(16u,estimate.x);
    smsDiagnosticMaxFloat(17u,estimate.y);
    smsDiagnosticMaxFloat(18u,estimate.z);
    smsDiagnosticMaxFloat(19u,asfloat(work.words[11].xyz).x+asfloat(work.words[11].xyz).y+asfloat(work.words[11].xyz).z);
    t_smsHdr[pixel]=float4(value,previous.w);
}
