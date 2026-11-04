// Native immutable producer addresses (24-byte CPU/Slang ABI). Prefix RGB is
// frozen independently of the root record. Its .w caches log2(luminance(F)), BEFORE
// reciprocal weighting; receiver-domain presence is the non-Absent work phase.
__exported import sms_work;
struct GiSmsContext {
    SmsWorkRecord* work;
    float4* prefixes;
    uint resolution,flags;
    SmsReservoir* temporalReservoir; // per-frame temporal merge output (read by reuse)
    SmsReservoir* finalReservoir;    // persistent last-frame post-reuse state (temporal input)
};
[[vk::binding(25,0)]] StructuredBuffer<GiSmsContext> t_smsProducer;

// Optional observation only, in a separate descriptor (ABI is unchanged).
// Readback is host-side after completion; never used for any rendering decision.
[[vk::binding(26,0)]] RWStructuredBuffer<uint> t_smsDiagnostics;
// Hong et al. 2025 section 5 temporal reprojection source. The motion vectors
// hold (prevPixel - curPixel) in pixels; the 1x1 dummy bound when unavailable
// reads as zero motion, which is exact for static-camera qualification scenes.
[[vk::binding(18,0)]] Texture2D<float2> t_smsMotionVectors;

float smsLuminance(float3 value) {
    return dot(value,float3(.2126,.7152,.0722));
}
SmsReservoir smsEmptyReservoir() {
    SmsReservoir reservoir;
    for (uint w=0u;w<16u;++w) reservoir.record.words[w]=uint4(0u,0u,0u,0u);
    reservoir.prefix=float4(0.0,0.0,0.0,0.0);
    reservoir.estimate=float3(0.0,0.0,0.0);
    reservoir.weight=asuint(0.0);
    reservoir.mass=asuint(0.0);
    reservoir.count=0u;
    reservoir.reserved0=0u;
    reservoir.reserved1=0u;
    reservoir.reserved2=0u;
    return reservoir;
}
bool smsReservoirValid(SmsReservoir reservoir) {
    float mass=asfloat(reservoir.mass);
    float weight=asfloat(reservoir.weight);
    return mass>0.0 && isfinite(mass) && weight>=0.0 && isfinite(weight) &&
        reservoir.count>=1u && reservoir.count<=2u &&
        reservoir.record.words[14].y==1u;
}
static const uint kSmsDiagEligible=0u,kSmsDiagRoots=1u,kSmsDiagCanonicalCandidates=2u;
static const uint kSmsDiagNeighborDomains=3u,kSmsDiagNeighborCandidates=4u;
static const uint kSmsDiagCanonicalAttempts=5u,kSmsDiagCanonicalAccepts=6u;
static const uint kSmsDiagNeighborAttempts=7u,kSmsDiagNeighborAccepts=8u;
static const uint kSmsDiagSelectedNeighbors=9u,kSmsDiagTopologyFailures=10u;
static const uint kSmsDiagSolverFailures=11u,kSmsDiagVisibilityFailures=12u;
static const uint kSmsDiagSupportFailures=13u,kSmsDiagMaxBaseRed=16u;
static const uint kSmsDiagMaxBaseGreen=17u,kSmsDiagMaxBaseBlue=18u;
static const uint kSmsDiagMaxFinalEstimate=19u,kSmsDiagMaxConditionalMass=20u;
static const uint kSmsDiagMaxReciprocalWeight=21u,kSmsDiagCount=32u;
bool smsDiagnosticsEnabled() { return (t_smsProducer[0].flags&2u)!=0u; }
void smsDiagnostic(uint index) {
    if (!smsDiagnosticsEnabled()) return;
    uint ignored;
    InterlockedAdd(t_smsDiagnostics[index],1u,ignored);
}
void smsDiagnosticMax(uint index,uint value) {
    if (!smsDiagnosticsEnabled()) return;
    uint ignored;
    InterlockedMax(t_smsDiagnostics[index],value,ignored);
}
// Positive finite floats compare monotonically by ordered uint bits.
void smsDiagnosticMaxFloat(uint index,float value) {
    if (!smsDiagnosticsEnabled() || !(value>0.0) || !isfinite(value)) return;
    uint bits=asuint(value);
    if ((bits&0x80000000u)!=0u) return;
    smsDiagnosticMax(index,bits);
}
