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
