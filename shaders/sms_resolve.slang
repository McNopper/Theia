// Local, unmerged SMS estimate. Spatial MIS will consume the completed local
// work records, never feed a recursively merged result back into this source.
import common;
import sms_work;
[[vk::binding(1,0)]] RWTexture2D<float4> t_smsHdr;
[[vk::push_constant]] SmsQueuePush pc;

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
    float3 value=previous.xyz+asfloat(work.words[11].xyz)*asfloat(work.words[14].x);
    if (!all(isfinite(value))) { smsQueueCompletionError(pc,kSmsQueueNumericError); return; }
    t_smsHdr[pixel]=float4(value,previous.w);
}
