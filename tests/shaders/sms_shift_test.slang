// 2d.2 failed-inverse + topology controls: the pure inverse-check contract of
// sms_shift.slang ("Every accepted map is inverse-checked; failed maps define
// zero MIS support"). The match goes through smsInverseRootMatches - the same
// entry point the shift chain uses - so a mutant of that check is observable.
// No scene or diagnostic-buffer dependency: standalone compute fixture.
import common;
import math;
import sms_geometry;
import sms_solver;
import sms_shift;

struct Input {
    float4 receiver; // xyz receiver position, w = mode
    float4 original; // xyz root position, w = etaTOverI
    float4 inverse;  // xyz root position, w = etaTOverI
    uint4 meta;      // x=original event, y=inverse event, z=original key, w=inverse key
};
struct Push {
    Input* inputs;
    float4* outputs;
    uint count, pad;
};
[[vk::push_constant]] Push pc;

void fillVertex(out SmsManifoldVertex vertex, float3 position, float eta, uint event, uint key) {
    vertex = (SmsManifoldVertex)0;
    vertex.position = position;
    vertex.etaTOverI = eta;
    vertex.event = event;
    vertex.objectKey = uint2(key, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    Input input = pc.inputs[id.x];
    uint mode = uint(input.receiver.w);
    float3 receiver = input.receiver.xyz;
    SmsManifoldVertex original[2], inverse[2];
    original[0] = original[1] = (SmsManifoldVertex)0;
    inverse[0] = inverse[1] = (SmsManifoldVertex)0;
    fillVertex(original[0], input.original.xyz, input.original.w, input.meta.x, input.meta.z);
    fillVertex(inverse[0], input.inverse.xyz, input.inverse.w, input.meta.y, input.meta.w);
    // The direction difference sameSmsRoot measures for the single root (both
    // lengths > 0 by construction of every case below).
    // The direction difference sameSmsRoot measures, with ONE shared length for
    // both divisions: this backend lowers GLSL Length() with per-call-site
    // precision divergence (two Length calls on identical vectors returned
    // different approximations), which would pollute the exact-zero oracle;
    // sameSmsRoot itself tolerates that at 1e-4.
    float3 da = original[0].position - receiver;
    float3 db = inverse[0].position - receiver;
    float l = length(da);
    float difference = l > 0.0 ? length(da / l - db / l) : 0.0;
    uint count = mode == 6u ? 0u : 1u;
    bool match = smsInverseRootMatches(receiver, original, inverse, count);
    bool topology = smsShiftTopology(original, inverse, count);
    pc.outputs[id.x] = float4(match ? 1.0 : 0.0, topology ? 1.0 : 0.0, difference, 1.0);
}
