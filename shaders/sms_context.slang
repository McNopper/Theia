// Native immutable producer addresses (24-byte CPU/Slang ABI). Prefix RGB is
// frozen independently of the root record. Its .w caches log2(luminance(F)), BEFORE
// reciprocal weighting; receiver-domain presence is the non-Absent work phase.
__exported import sms_work;
struct GiSmsContext {
    SmsWorkRecord* work;
    float4* prefixes;
    uint resolution,flags;
};
[[vk::binding(25,0)]] StructuredBuffer<GiSmsContext> t_smsProducer;

// Optional observation only, in a separate descriptor (ABI is unchanged).
// Readback is host-side after completion; never used for any rendering decision.
[[vk::binding(26,0)]] RWStructuredBuffer<uint> t_smsDiagnostics;
static const uint kSmsDiagEligible=0u,kSmsDiagRoots=1u,kSmsDiagCanonicalCandidates=2u;
static const uint kSmsDiagNeighborDomains=3u,kSmsDiagNeighborCandidates=4u;
static const uint kSmsDiagCanonicalAttempts=5u,kSmsDiagCanonicalAccepts=6u;
static const uint kSmsDiagNeighborAttempts=7u,kSmsDiagNeighborAccepts=8u;
static const uint kSmsDiagSelectedNeighbors=9u,kSmsDiagTopologyFailures=10u;
static const uint kSmsDiagSolverFailures=11u,kSmsDiagVisibilityFailures=12u;
static const uint kSmsDiagSupportFailures=13u,kSmsDiagCount=16u;
bool smsDiagnosticsEnabled() { return (t_smsProducer[0].flags&2u)!=0u; }
void smsDiagnostic(uint index) {
    if (!smsDiagnosticsEnabled()) return;
    uint ignored;
    InterlockedAdd(t_smsDiagnostics[index],1u,ignored);
}
