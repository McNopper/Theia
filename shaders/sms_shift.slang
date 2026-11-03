// Hong 2025 Section 5.2--5.3: delta-root / static-light-area shift, J=1.
// Every accepted map is inverse-checked; failed maps define zero MIS support.
import common;
import math;
import sms_transport;
import sms_scene;

bool smsShiftTopology(SmsManifoldVertex source[2],SmsManifoldVertex target[2],uint count) {
    if (count<1u || count>2u) return false;
    for (uint i=0u;i<count;++i)
        if (any(source[i].objectKey!=target[i].objectKey) || source[i].event!=target[i].event ||
            source[i].etaTOverI!=target[i].etaTOverI) return false;
    return true;
}
bool smsShiftOnce(inout TheiaSmsScene scene,float3 receiver,uint remainingDepth,float3 light,
    SmsManifoldVertex source[2],uint count,out SmsManifoldVertex target[2]) {
    target[0]=target[1]=(SmsManifoldVertex)0;
    if (count<1u || count>2u || remainingDepth<count+2u) return false;
    if (!scene.project(receiver,source[0].position,source,count,target) ||
        !smsShiftTopology(source,target,count)) return false;
    // This is a bounded SHIFT walk, not a cap on Bernoulli estimator trials.
    if (!solveSmsManifold(scene,receiver,light,target,count,5u,1e-5)) return false;
    float3 edge=light-target[count-1u].position;
    float distance=length(edge);
    if (!(distance>kRayEpsilon+kMediumEpsilon) || !isfinite(distance)) return false;
    return !scene.occluded(target[count-1u].position,edge/distance,distance-kRayEpsilon);
}
bool smsInverseRootMatches(float3 receiver,SmsManifoldVertex original[2],SmsManifoldVertex inverse[2],uint count) {
    return sameSmsRoot(receiver,original,count,inverse,count,kSmsRootDirectionTolerance);
}
bool smsShiftInverseChecked(inout TheiaSmsScene scene,float3 sourceReceiver,uint sourceDepth,
    float3 targetReceiver,uint targetDepth,float3 light,SmsManifoldVertex original[2],uint count,
    out SmsManifoldVertex target[2]) {
    if (!smsShiftOnce(scene,targetReceiver,targetDepth,light,original,count,target)) return false;
    SmsManifoldVertex inverse[2];
    if (!smsShiftOnce(scene,sourceReceiver,sourceDepth,light,target,count,inverse)) return false;
    return smsInverseRootMatches(sourceReceiver,original,inverse,count);
}
