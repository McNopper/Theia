// Native Theia scene adapter for the common SMS sampler. These are GiPass's
// existing set-0 scene bindings, not Hyperion's descriptor/instance layout.
import common;
import math;
import surface;
import sms_transport;

[[vk::binding(0,0)]] RaytracingAccelerationStructure t_smsTlas;
[[vk::binding(6,0)]] StructuredBuffer<GpuMaterial> t_smsMaterials;
[[vk::binding(7,0)]] StructuredBuffer<GpuVertex> t_smsVertices;
[[vk::binding(8,0)]] StructuredBuffer<GpuInstance> t_smsInstances;
[[vk::binding(9,0)]] StructuredBuffer<uint> t_smsIndices;
[[vk::binding(10,0)]] StructuredBuffer<GpuEmissiveTriangle> t_smsEmitters;
[[vk::binding(13,0)]] Sampler2D<float4> t_smsTextures[];
[[vk::binding(14,0)]] StructuredBuffer<float> t_smsEmitterCdf;
[[vk::binding(19,0)]] StructuredBuffer<float4x4> t_smsInstanceTransforms;

struct TheiaSmsHit {
    SmsManifoldVertex vertex;
    uint primitive;
    float distance;
};

float3 theiaSmsUnitDerivative(float3 normal,float magnitude,float3 derivative) {
    return (derivative-normal*dot(normal,derivative))/magnitude;
}

struct TheiaSmsTextureSampler : ISurfaceTextureSampler {
    float4 sampleTex(uint index,float2 uv) {
        return t_smsTextures[NonUniformResourceIndex(index)].SampleLevel(uv,0.0);
    }
};

struct TheiaSmsScene : ISmsProjector {
    bool occluded(float3 origin,float3 direction,float maximum) {
        RayDesc ray;
        ray.Origin=origin; ray.Direction=direction;
        ray.TMin=kMediumEpsilon; ray.TMax=maximum;
        RayQuery<RAY_FLAG_FORCE_OPAQUE|RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES|RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(t_smsTlas,RAY_FLAG_NONE,kInstanceMaskAll,ray);
        while (query.Proceed()) {}
        return query.CommittedStatus()==COMMITTED_TRIANGLE_HIT;
    }
    bool trace(float3 origin,float3 direction,float maximum,out TheiaSmsHit hit) {
        hit={};
        RayDesc ray;
        ray.Origin=origin; ray.Direction=direction;
        ray.TMin=kMediumEpsilon; ray.TMax=maximum;
        // Theia builds triangle BLASes, including tessellated sphere meshes.
        // Alpha surfaces block this supported domain rather than being skipped.
        RayQuery<RAY_FLAG_FORCE_OPAQUE|RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(t_smsTlas,RAY_FLAG_NONE,kInstanceMaskAll,ray);
        while (query.Proceed()) {}
        if (query.CommittedStatus()!=COMMITTED_TRIANGLE_HIT) return false;
        uint instanceId=query.CommittedInstanceID();
        GpuInstance instance=t_smsInstances[instanceId];
        hit.primitive=query.CommittedPrimitiveIndex();
        hit.distance=query.CommittedRayT();
        uint first=instance.indexOffset+hit.primitive*3u;
        if (first+2u>=instance.indexOffset+instance.indexCount) return false;
        GpuVertex a=t_smsVertices[t_smsIndices[first]],b=t_smsVertices[t_smsIndices[first+1u]],
            c=t_smsVertices[t_smsIndices[first+2u]];
        float4x4 o2w=t_smsInstanceTransforms[instanceId];
        float3x3 normalMatrix=transpose(inverse3x3((float3x3)o2w));
        float3 p0=mul(o2w,float4(a.position,1.0)).xyz;
        float3 p1=mul(o2w,float4(b.position,1.0)).xyz,p2=mul(o2w,float4(c.position,1.0)).xyz;
        float3 e1=p1-p0,e2=p2-p0,crossEdges=cross(e1,e2);
        float area2=length(crossEdges);
        if (!(area2>0.0)) return false;
        float3 n0=mul(normalMatrix,a.normal),n1=mul(normalMatrix,b.normal),n2=mul(normalMatrix,c.normal);
        n0=normalize(n0); n1=normalize(n1); n2=normalize(n2);
        float2 bary=query.CommittedTriangleBarycentrics();
        float3 weights=float3(1.0-bary.x-bary.y,bary.x,bary.y);
        SmsManifoldVertex vertex={};
        // Use the committed ray parameter for position. Barycentric interpolation
        // of large world vertices can quantize the Newton projection enough to
        // prevent convergence; barycentrics still interpolate the shading normal.
        vertex.position=origin+direction*hit.distance;
        vertex.objectKey=uint2(instanceId,instance.materialIndex);
        vertex.etaTOverI=1.0;
        vertex.geoNormal=crossEdges/area2; // raw outward winding; never ray-facing
        vertex.dpdu=normalize(e1); vertex.dpdv=cross(vertex.geoNormal,vertex.dpdu);
        float3 raw=n0*weights.x+n1*weights.y+n2*weights.z;
        // Match reconstructTriangleGeometry's branch, including equality and
        // unordered/NaN blends from zero per-vertex normalization. Do not remove
        // real intersections or invent a different normal field for SMS.
        if (dot(raw,raw)>1.0e-8) {
            float magnitude=length(raw);
            float sign=dot(raw,vertex.geoNormal)<0.0 ? -1.0 : 1.0;
            raw*=sign; vertex.normal=raw/magnitude;
            float d11=dot(e1,e1),d22=dot(e2,e2),d12=dot(e1,e2),determinant=dot(crossEdges,crossEdges);
            float3 g1=(d22*e1-d12*e2)/determinant,g2=(d11*e2-d12*e1)/determinant;
            float3 dn1=(n1-n0)*sign,dn2=(n2-n0)*sign;
            vertex.dndu=theiaSmsUnitDerivative(vertex.normal,magnitude,
                dn1*dot(g1,vertex.dpdu)+dn2*dot(g2,vertex.dpdu));
            vertex.dndv=theiaSmsUnitDerivative(vertex.normal,magnitude,
                dn1*dot(g1,vertex.dpdv)+dn2*dot(g2,vertex.dpdv));
        } else {
            vertex.normal=vertex.geoNormal;
            vertex.dndu=vertex.dndv=0.0.xxx;
        }
        hit.vertex=vertex;
        return all(isfinite(vertex.position)) && all(isfinite(vertex.normal)) &&
            all(isfinite(vertex.dndu)) && all(isfinite(vertex.dndv));
    }

    bool supportedEvent(float3 previous,inout SmsManifoldVertex vertex,uint event) {
        GpuMaterial material=t_smsMaterials[vertex.objectKey.y];
        bool front=dot(previous-vertex.position,vertex.geoNormal)>0.0;
        if (event==kSmsReflection) {
            if (!smsReflectionCaster(material) || !front) return false;
            vertex.etaTOverI=1.0;
        } else {
            if (!smsSmoothGlass(material)) return false;
            float eta=smsGlassEta(material);
            if (!(eta>1.0)) return false;
            vertex.etaTOverI=front ? eta : 1.0/eta;
            if (!front && smsPureAbsorption(material)) {
                vertex.normal=vertex.geoNormal;
                vertex.dndu=vertex.dndv=0.0.xxx; // the medium walk uses triangle geometric normals
            }
        }
        vertex.event=event;
        return true;
    }

    [mutating]
    bool project(float3 receiver,float3 proposedFirst,SmsManifoldVertex reference[2],
        uint count,out SmsManifoldVertex projected[2]) {
        projected[0]=projected[1]=(SmsManifoldVertex)0;
        float3 edge=proposedFirst-receiver;
        if (!(dot(edge,edge)>0.0)) return false;
        TheiaSmsHit first;
        if (!trace(receiver,normalize(edge),kInf,first)) return false;
        projected[0]=first.vertex;
        if (!supportedEvent(receiver,projected[0],reference[0].event)) return false;
        if (count==2u) {
            if (projected[0].event!=kSmsTransmission || projected[0].etaTOverI<=1.0) return false;
            float3 wo=normalize(receiver-projected[0].position),normal=projected[0].normal;
            if (dot(normal,wo)<0.0) normal=-normal;
            float3 direction; float weight;
            if (!smsDeltaInterface(wo,normal,kSmsTransmission,1.0/projected[0].etaTOverI,direction,weight)) return false;
            TheiaSmsHit second;
            if (!trace(projected[0].position,direction,kInf,second)) return false;
            projected[1]=second.vertex;
            if (!supportedEvent(projected[0].position,projected[1],kSmsTransmission) ||
                any(projected[0].objectKey!=projected[1].objectKey) || projected[1].etaTOverI>=1.0) return false;
        }
        return true;
    }
};
