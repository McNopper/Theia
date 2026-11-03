import sms_resampling;
struct Push { uint4* inputs; float4* outputs; uint count,pad; };
[[vk::push_constant]] Push pc;
float3 integrand(uint root) {
    return root==0u ? float3(2,1,3) : root==1u ? float3(7,4,1) : float3(1,5,6);
}
float proxy(uint domain,uint root,bool alternative) {
    float value=domain==0u ? (root==0u?2.0:root==1u?9.0:5.0) :
        domain==1u ? (root==0u?8.0:root==1u?1.0:4.0) : (root==0u?3.0:11.0);
    return value*(alternative ? (domain==0u?1.5:domain==1u?1.0/7.0:9.0) : 1.0);
}
float proposal(uint domain,uint local) {
    return domain==0u ? (local==0u?.5:local==1u?1.0/3.0:1.0/6.0) :
        domain==1u ? (local==0u?1.0/3.0:local==1u?.5:1.0/6.0) : (local==0u?.1:.9);
}

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x>=pc.count) return;
    uint4 input=pc.inputs[id.x];
    if (input.x==7u) {
        SmsLogResamplingState state=emptySmsLogResampling();
        bool ok=true;
        if (input.y==1u) ok=smsLogResamplingAdd(state,0u,-120.0,0.0,0.0,0.0);
        ok=ok && smsLogResamplingAdd(state,input.y,-20.0,0.0,0.0,asfloat(0x3F7FFFFFu));
        float3 color=kSmsLogZero.xxx,estimate=0.0.xxx;
        if (state.selected==0u) color.x=input.y==0u?-20.0:-120.0;
        if (state.selected==1u) color.y=-20.0;
        ok=ok && smsLogResamplingEstimate(state,color,estimate);
        pc.outputs[id.x]=float4(float(state.selected),state.weightSum,estimate.xy*1048576.0); return;
    }
    if (input.x==4u) {
        float cached=7.0;
        if (input.y==3u) {
            bool bypass=smsCacheLocalTarget(false,float3(asfloat(0x7F800000u),0,0),asfloat(0x7FC00000u),cached);
            pc.outputs[id.x]=float4(bypass?1.0:0.0,cached,0,0); return;
        }
        float weighted=input.y==0u?1e38:input.y==1u?1e-20:1e8;
        float pdf=input.y==0u?4.0:input.y==1u?1e-30:1e-38;
        float count=input.y==0u?1.0:4.0;
        float3 color=kSmsLogZero.xxx; float ucwLog=kSmsLogZero;
        bool ok=smsCacheLocalTarget(true,float3(weighted,0,0),pdf,cached) &&
            smsLogContribution(float3(weighted,0,0),1.0.xxx,1.0,1.0.xxx,1.0,pdf.xxx,color) &&
            smsLogUnbiasedWeight(count,1.0,pdf,ucwLog);
        SmsLogResamplingState state=emptySmsLogResampling();
        ok=ok && smsLogResamplingAdd(state,0u,cached,ucwLog,0.0,0.0);
        float3 estimate=0.0.xxx;
        ok=ok && smsLogResamplingEstimate(state,color,estimate);
        float physical=weighted*pdf,ucw=count/pdf;
        bool legacyFailure=input.y==0u?!isfinite(physical):input.y==1u?physical==0.0:!isfinite(ucw);
#ifdef SMS_LEGACY_RANGE_CONTROL
        ok=ok && isfinite(physical) && physical>0.0 && isfinite(ucw);
#endif
        pc.outputs[id.x]=float4(ok?estimate.x:0.0,
            isfinite(cached) && cached!=kSmsLogZero?1.0:0.0,legacyFailure?1.0:0.0,ok?1.0:0.0);
        return;
    }
    if (input.x==6u) {
        float u=input.y==0u?0.0:input.y==1u?.49999:input.y==2u?.5:.99999;
        SmsLogResamplingState state=emptySmsLogResampling();
        bool ok=smsLogResamplingAdd(state,0u,0.0,0.0,0.0,0.0) &&
            smsLogResamplingAdd(state,1u,0.0,0.0,0.0,u);
        float3 integrand=kSmsLogZero.xxx,estimate=0.0.xxx;
        ok=ok && smsLogRgb(state.selected==0u?float3(1,0,0):float3(0,1,0),integrand) &&
            smsLogResamplingEstimate(state,integrand,estimate);
        pc.outputs[id.x]=float4(ok?float(state.selected):-1.0,exp2(state.weightSum),estimate.xy); return;
    }
    if (input.x==5u) {
        float scales[3]={0,0,0};
        if (input.y==1u) { scales[0]=log2(1.5); scales[1]=-log2(7.0); scales[2]=log2(9.0); }
        if (input.y==2u) { scales[0]=-40.0*log2(10.0); scales[1]=38.0*log2(10.0); scales[2]=-30.0*log2(10.0); }
        if (input.y==3u) { scales[0]=38.0*log2(10.0); scales[1]=-40.0*log2(10.0); scales[2]=30.0*log2(10.0); }
        float3 expectation=0.0.xxx; uint invalid=0u;
        for (uint a=0u;a<3u;++a) for (uint b=0u;b<3u;++b) for (uint d=0u;d<2u;++d) {
            uint roots[3]={a,b==2u?3u:b,d==0u?0u:2u};
            float targets[3],weights[3];
            float probability=proposal(0u,a)*proposal(1u,b)*proposal(2u,d);
            SmsLogResamplingState reservoir=emptySmsLogResampling();
            float total=kSmsLogZero;
            for (uint i=0u;i<3u;++i) {
                weights[i]=targets[i]=kSmsLogZero;
                if (i==1u && b==2u) continue;
                uint root=roots[i];
                float target=log2(proxy(0u,root,false))+scales[0],mis=kSmsLogZero;
                if (i==0u) {
                    for (uint j=1u;j<3u;++j) {
                        bool support=j==1u?root!=2u:root!=1u;
                        float part,unused,source=log2(proxy(j,root,false))+scales[j];
                        if (!smsLogPairwiseMis(target,source,2u,support,part,unused)) ++invalid;
                        mis=smsLogAdd(mis,part);
                    }
                } else {
                    float unused,source=log2(proxy(i,root,false))+scales[i];
                    if (!smsLogPairwiseMis(target,source,2u,true,unused,mis)) ++invalid;
                }
                float q=proposal(i,i==0u?a:i==1u?b:d);
                targets[i]=target; weights[i]=target-log2(q)+mis; total=smsLogAdd(total,weights[i]);
                if (!smsLogResamplingAdd(reservoir,i,target,-log2(q),mis,.37)) ++invalid;
            }
            if (abs(reservoir.weightSum-total)>1e-5) ++invalid;
            for (uint i=0u;i<3u;++i) {
                if (weights[i]==kSmsLogZero) continue;
                SmsLogResamplingState selected=reservoir;
                selected.selected=i; selected.selectedTarget=targets[i];
                float3 logarithm=kSmsLogZero.xxx,estimate=0.0.xxx;
                if (!smsLogRgb(integrand(roots[i]),logarithm) || !smsLogResamplingEstimate(selected,logarithm,estimate)) ++invalid;
                expectation+=probability*exp2(weights[i]-total)*estimate;
            }
        }
        pc.outputs[id.x]=float4(expectation,float(invalid)); return;
    }
    if (input.x==2u) {
        float canonical,neighbor;
        bool ok=smsPairwiseMis(1e38,1e38,3u,true,canonical,neighbor);
        float c,n;
        bool rejected=!smsPairwiseMis(-1.0,1.0,2u,true,c,n) &&
            !smsPairwiseMis(asfloat(0x7FC00000u),1.0,2u,true,c,n) &&
            !smsPairwiseMis(1.0,asfloat(0x7F800000u),2u,true,c,n);
        SmsResamplingState state=emptySmsResampling();
        bool overflow=!smsResamplingAdd(state,0u,3e38,3e38,1.0,0.0);
        pc.outputs[id.x]=float4(ok?canonical:-1.0,neighbor,rejected?1.0:0.0,overflow?1.0:0.0);
        return;
    }
    if (input.x==3u) {
        float u=input.y==0u ? 0.0 : input.y==1u ? .74999 : input.y==2u ? .75 : .99999;
        SmsResamplingState state=emptySmsResampling();
        bool ok=smsResamplingAdd(state,0u,1.0,1.0,1.0,0.0);
        ok=ok && smsResamplingAdd(state,1u,1.0,3.0,1.0,u);
        float3 color=0.0.xxx;
        ok=ok && smsResamplingEstimate(state,state.selected==0u ? float3(1,0,0) : float3(0,1,0),color);
        pc.outputs[id.x]=float4(ok?float(state.selected):-1.0,state.weightSum,color.xy);
        return;
    }
    bool alternative=input.x==1u;
    float3 expectation=0.0.xxx;
    uint invalid=0u;
    for (uint a=0u;a<3u;++a) for (uint b=0u;b<3u;++b) for (uint d=0u;d<2u;++d) {
        uint roots[3]={a,b==2u?3u:b,d==0u?0u:2u};
        float targets[3],weights[3];
        float probability=proposal(0u,a)*proposal(1u,b)*proposal(2u,d);
        SmsResamplingState reservoir=emptySmsResampling();
        float total=0.0;
        for (uint i=0u;i<3u;++i) {
            weights[i]=targets[i]=0.0;
            if (i==1u && b==2u) continue; // sampled source lies outside its invertible shift support
            uint root=roots[i];
            float target=proxy(0u,root,alternative),mis=0.0;
            if (i==0u) {
                for (uint j=1u;j<3u;++j) {
                    bool support=j==1u ? root!=2u : root!=1u;
                    float part,unused;
                    if (!smsPairwiseMis(target,proxy(j,root,alternative),2u,support,part,unused)) ++invalid;
                    mis+=part;
                }
            } else {
                float unused;
                if (!smsPairwiseMis(target,proxy(i,root,alternative),2u,true,unused,mis)) ++invalid;
            }
            float q=proposal(i,i==0u?a:i==1u?b:d);
            targets[i]=target; weights[i]=mis*target/q; total+=weights[i];
            if (!smsResamplingAdd(reservoir,i,target,1.0/q,mis,.37)) ++invalid;
        }
        if (abs(reservoir.weightSum-total)>1e-5*max(1.0,total)) ++invalid;
        if (total==0.0) continue;
        for (uint i=0u;i<3u;++i) {
            if (weights[i]==0.0) continue;
            SmsResamplingState selected=reservoir;
            selected.selected=i; selected.selectedTarget=targets[i];
            float3 estimate;
            if (!smsResamplingEstimate(selected,integrand(roots[i]),estimate)) ++invalid;
            expectation+=probability*(weights[i]/total)*estimate;
        }
    }
    pc.outputs[id.x]=float4(expectation,float(invalid));
}
