// Native immutable producer addresses (24-byte CPU/Slang ABI). Prefix RGB is
// frozen independently of the root record. Its .w caches luminance(F), BEFORE
// reciprocal weighting; receiver-domain presence is the non-Absent work phase.
__exported import sms_work;
struct GiSmsContext {
    SmsWorkRecord* work;
    float4* prefixes;
    uint resolution,flags;
};
[[vk::binding(25,0)]] StructuredBuffer<GiSmsContext> t_smsProducer;
