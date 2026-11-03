import sms_scene;
import common;
import math;
import sms_transport;
struct Input { float4 origin,direction,light; };
struct Output { float4 position,normal,geometric,dpdu,dpdv,dndu,dndv,keys; };
struct Push { Input* inputs; Output* outputs; uint count,mode; };
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x>=pc.count) return;
    Input input=pc.inputs[id.x];
    Output result={};
    TheiaSmsScene scene;
    if (pc.mode==3u) {
        result.position.w=scene.occluded(input.origin.xyz,normalize(input.direction.xyz),input.light.w) ? 1.0 : 0.0;
        pc.outputs[id.x]=result; return;
    }
    TheiaSmsHit hit;
    if (!scene.trace(input.origin.xyz,normalize(input.direction.xyz),kInf,hit)) {
        pc.outputs[id.x]=result; return;
    }
    SmsManifoldVertex vertex=hit.vertex;
    if (pc.mode!=0u) {
        uint count=pc.mode==1u ? 1u : 2u;
        SmsManifoldVertex reference[2],root[2];
        reference[0]=vertex; reference[1]=(SmsManifoldVertex)0;
        reference[0].event=count==1u ? kSmsReflection : kSmsTransmission;
        if (!scene.project(input.origin.xyz,vertex.position,reference,count,root) ||
            !solveSmsManifold(scene,input.origin.xyz,input.light.xyz,root,count,32u,1e-5)) {
            pc.outputs[id.x]=result; return;
        }
        float geometric,residual;
        if (!smsGeometricTerm(input.origin.xyz,root,count,input.light.xyz,float3(1,0,0),float3(0,1,0),geometric) ||
            !smsManifoldResidual(input.origin.xyz,root,count,input.light.xyz,residual)) {
            pc.outputs[id.x]=result; return;
        }
        result.position=float4(root[0].position,float(count));
        result.geometric=float4(root[1].position,geometric);
        result.keys=float4(float2(root[0].objectKey),float(count),residual);
        pc.outputs[id.x]=result; return;
    }
    result.position=float4(vertex.position,1);
    result.normal=float4(vertex.normal,0); result.geometric=float4(vertex.geoNormal,0);
    result.dpdu=float4(vertex.dpdu,0); result.dpdv=float4(vertex.dpdv,0);
    result.dndu=float4(vertex.dndu,0); result.dndv=float4(vertex.dndv,0);
    result.keys=float4(float2(vertex.objectKey),float(hit.primitive),hit.distance);
    pc.outputs[id.x]=result;
}
