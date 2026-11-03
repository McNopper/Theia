// Included after GiHit / tracer / medium / surface helpers. One independent
// frozen receiver per pixel; ordinary PT candidates never write this storage.
struct GiSmsContext {
    SmsWorkRecord* work;
    float4* prefixes;
    uint resolution,pad;
};
[[vk::binding(25,0)]] StructuredBuffer<GiSmsContext> t_smsProducer;

void initializeSmsPixel(uint2 pixel) {
    if (smsEnabled==0u) return;
    GiSmsContext context=t_smsProducer[0];
    uint slot=pixel.y*screenWidth+pixel.x;
    SmsWorkRecord work={};
    work.words[7].zw=uint2(hashCombine(slot,rngBaseSeed^0x53534D53u),hashCombine(frameSampleIndex,0x534D5350u));
    work.words[12].w=context.resolution;
    work.words[13].zw=uint2(emissiveTriangleCount,maxDepth);
    work.words[14]=uint4(asuint(exposure),1u,frameSampleIndex,rngBaseSeed);
    work.words[15].zw=uint2(screenWidth,screenHeight);
    context.work[slot]=work;
    context.prefixes[slot]=0.0.xxxx;
}

void failSmsPixel(uint slot) {
    // The common preparation/validation pass rejects this explicit invalid phase;
    // no incomplete/numeric-error sample can silently become an absent estimate.
    t_smsProducer[0].work[slot].words[1].w=0xFFFFFFFFu;
}

void produceSmsReceiver(uint2 pixel,GiHit initialHit,SurfaceHit initialSurface) {
    if (smsEnabled==0u) return;
    GiSmsContext context=t_smsProducer[0];
    uint slot=pixel.y*screenWidth+pixel.x;
    GiHit hit=initialHit;
    SurfaceHit surface=initialSurface;
    PathState state={};
    state.throughput=1.0.xxx;
    state.seed=composeRngSeed(pixel,frameSampleIndex,0x53D5u,rngBaseSeed);
    state.mediumEta=1.0;
    SmsPathState path=beginSmsPath();
    TheiaTracer tracer;
    for (uint depth=0u;depth<maxDepth && !state.terminated;++depth) {
        uint remaining=maxDepth-depth;
        if (!all(isfinite(state.throughput)) || any(state.throughput<0.0.xxx) ||
            !all(isfinite(surface.mat.baseColorWeight)) || !all(isfinite(surface.mat.emissionColorLum))) {
            failSmsPixel(slot); return;
        }
        if (smsCapturePathReceiver(path,surface.mat,remaining)) {
            SmsWorkRecord work=context.work[slot];
            work.words[0]=uint4(asuint(hit.worldPos),hit.materialIdx);
            work.words[1]=uint4(asuint(hit.shadingNormal),kSmsPrepare);
            work.words[2]=uint4(asuint(hit.geoNormal),asuint(hit.uv.x));
            work.words[3]=uint4(asuint(hit.tangent),asuint(hit.uv.y));
            work.words[4]=uint4(asuint(surface.wo),slot);
            work.words[9].xyz=asuint(state.throughput);
            work.words[13].w=remaining;
            context.prefixes[slot]=float4(state.throughput,1);
            context.work[slot]=work;
            return;
        }
        if (!path.prefixSpecular || remaining<3u) return;
        state.depth=depth;
        integrateSurface(tracer,surface,emissiveTriangleCount,emissiveTriangles,emissiveCdf,
            envMarginalCdf,envConditionalCdf,envImportanceWidth,envImportanceHeight,hasEnvMap!=0u,
            envLuminanceScale,t_envMap,s_env,state);
        smsAdvancePathSurface(path,surface,state,hit.geoNormal,uint2(hit.instanceIndex,hit.materialIdx),remaining,false);
        if (state.terminated || !path.prefixSpecular) return;
        float3 origin=state.nextOrigin,direction=state.nextDirection;
        if (state.insideMedium) runMediumWalk(state,origin,direction,path);
        if (state.terminated || !path.prefixSpecular) return;
        hit=traceClosestHit(origin,direction,kInf,kRayEpsilon);
        if (!hit.hit) return;
        surface=makeSurfaceHit(hit,-direction);
    }
}

bool smsGiCollectableEmitter(GiHit hit,SurfaceHit surface) {
    return smsCollectablePathEmitter(emissiveTriangleCount,emissiveTriangles,emissiveCdf,surface.mat,
        uint2(hit.instanceIndex,hit.materialIdx),hit.primitiveIndex,surface.wo,hit.geoNormal);
}
