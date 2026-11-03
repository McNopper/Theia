// Included by the native resolve after the common radiometric helpers. Inputs
// remain immutable/unmerged; only the final HDR pixel receives resampled energy.
static const uint kSmsSpatialNeighbors=3u;
float smsReuseTarget(float3 value) { return dot(value,float3(.2126,.7152,.0722)); }

bool smsLocalCandidate(SmsWorkRecord work,float proxy,out bool candidate,out float ucw) {
    candidate=false; ucw=0.0;
    uint phase=smsPhase(work);
    if (phase==kSmsAbsent) return true;
    if (phase!=kSmsDone || work.words[14].y!=1u || !isfinite(proxy) || proxy<0.0) return false;
    uint status=work.words[12].z,count=work.words[10].z;
    if (status==0u) return count==0u;
    if (status!=1u || count<1u || count>2u || !(proxy>0.0)) return false;
    SmsReciprocal reciprocal;
    reciprocal.trials=work.words[12].xy; reciprocal.status=status;
    float rootWeight;
    if (!completedSmsReciprocal(reciprocal,rootWeight)) return false;
    float mass=asfloat(work.words[11].w),lightPdf=asfloat(work.words[5].w);
    if (!(mass>0.0) || !isfinite(mass) || !(lightPdf>0.0) || !isfinite(lightPdf)) return false;
    ucw=smsConditionalWeight(rootWeight,mass)/lightPdf;
    if (!(ucw>0.0) || !isfinite(ucw)) return false;
    candidate=true;
    return true;
}

bool smsShiftIntegrand(inout TheiaSmsScene scene,SmsWorkRecord receiver,float3 prefix,
    SmsWorkRecord emitter,SmsManifoldVertex root[2],uint count,out float3 value) {
    SmsWorkRecord work=receiver;
    work.words[5]=emitter.words[5]; work.words[5].w=asuint(1.0); // physical F, not F/p_light
    work.words[6]=emitter.words[6]; work.words[7].xy=emitter.words[7].xy;
    work.words[9].xyz=asuint(prefix);
    return contribution(work,root,count,value);
}

bool smsSpatialEstimate(uint slot,uint width,uint height,SmsWorkRecord canonical,out float3 estimate) {
    estimate=0.0.xxx;
    if (smsPhase(canonical)==kSmsAbsent) return true;
    GiSmsContext context=t_smsProducer[0];
    float4 prefix=context.prefixes[slot];
    bool localCandidate; float localUcw;
    if (!smsLocalCandidate(canonical,prefix.w,localCandidate,localUcw)) return false;
    SmsManifoldVertex original[2]; uint originalCount=0u;
    if (localCandidate) smsLoadRoot(canonical,original,originalCount);
    float3 values[4]; float targets[4],ucws[4],weights[4];
    for (uint i=0u;i<4u;++i) { values[i]=0.0.xxx; targets[i]=ucws[i]=weights[i]=0.0; }
    if (localCandidate) {
        values[0]=asfloat(canonical.words[11].xyz)/localUcw;
        targets[0]=prefix.w; ucws[0]=localUcw;
    }
    uint2 pixel=uint2(slot%width,slot/width);
    uint selectedSlots[3];
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
        }
        float proxyAtCanonical=0.0;
        bool canonicalSupport=false;
        if (localCandidate && domain) {
            SmsManifoldVertex shifted[2];
            canonicalSupport=smsShiftInverseChecked(scene,smsReceiverPosition(canonical),canonical.words[13].w,
                smsReceiverPosition(source),source.words[13].w,smsLightPosition(canonical),original,originalCount,shifted);
            if (canonicalSupport) {
                float3 sourceValue;
                if (!smsShiftIntegrand(scene,source,sourcePrefix.xyz,canonical,shifted,originalCount,sourceValue)) return false;
                proxyAtCanonical=smsReuseTarget(sourceValue);
            }
        }
        if (localCandidate) {
            float part,unused;
            if (!smsPairwiseMis(prefix.w,proxyAtCanonical,kSmsSpatialNeighbors,canonicalSupport,part,unused)) return false;
            weights[0]+=part;
        }
        if (!domain || !sourceCandidate) continue;
        SmsManifoldVertex sourceRoot[2],shifted[2]; uint count;
        smsLoadRoot(source,sourceRoot,count);
        if (!smsShiftInverseChecked(scene,smsReceiverPosition(source),source.words[13].w,
            smsReceiverPosition(canonical),canonical.words[13].w,smsLightPosition(source),sourceRoot,count,shifted)) continue;
        float3 targetValue;
        if (!smsShiftIntegrand(scene,canonical,prefix.xyz,source,shifted,count,targetValue)) return false;
        float target=smsReuseTarget(targetValue),unused,mis;
        if (!smsPairwiseMis(target,sourcePrefix.w,kSmsSpatialNeighbors,true,unused,mis)) return false;
        values[i+1u]=targetValue; targets[i+1u]=target; ucws[i+1u]=sourceUcw; weights[i+1u]=mis;
    }
    SmsResamplingState reservoir=emptySmsResampling();
    float4 selection=smsTrialSample(canonical.words[7].zw,uint2(0,0),uint2(0xFFFFFFFBu,0));
    for (uint i=0u;i<4u;++i)
        if (!smsResamplingAdd(reservoir,i,targets[i],ucws[i],weights[i],selection[i])) return false;
    if (reservoir.weightSum==0.0) return true;
    if (reservoir.selected>=4u) return false;
    return smsResamplingEstimate(reservoir,values[reservoir.selected],estimate);
}
