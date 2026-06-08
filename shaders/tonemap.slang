// Tone-mapping fragment shader for the Theia real-time renderer.
// Copied from Hyperion. All colour maths in linear Rec.2020.

[[vk::binding(1, 0)]] RWTexture2D<float4> hdrTarget;

struct PushConstants {
    uint  frameIndex;
    uint  maxDepth;
    uint  rngSeed;
    float envLuminanceScale;
    uint  lightCount;
    uint  outputColorSpace;
    uint  samplesPerPixel;
    uint  hasEnvMap;
    uint  emissiveTriangleCount;
    uint  envImportanceWidth;
    uint  envImportanceHeight;
    uint  tonemapper;
};
[[vk::push_constant]] PushConstants pushConst;

float3 rec2020ToRec709(float3 c) {
    return float3(
        dot(c, float3( 1.6605f, -0.5876f, -0.0728f)),
        dot(c, float3(-0.1246f,  1.1329f, -0.0083f)),
        dot(c, float3(-0.0182f, -0.1006f,  1.1187f))
    );
}

float3 rec2020ToP3(float3 c) {
    return float3(
        dot(c, float3( 1.2249f, -0.2247f,  0.0000f)),
        dot(c, float3(-0.0420f,  1.0419f,  0.0000f)),
        dot(c, float3( 0.0000f, -0.0197f,  1.0197f))
    );
}

static const float3x3 kRec709ToP3 = float3x3(
    float3(0.822462f, 0.177538f, 0.000000f),
    float3(0.033194f, 0.966806f, 0.000000f),
    float3(0.017083f, 0.072340f, 0.910577f)
);

static const float3x3 kRec2020ToAP1 = float3x3(
    float3(0.6131324f,  0.3395255f,  0.0474491f),
    float3(0.0701243f,  0.9163394f,  0.0135363f),
    float3(0.0205076f,  0.1096098f,  0.8699926f)
);
static const float3x3 kAP1ToRec709 = float3x3(
    float3( 2.9090f, -1.6933f, -0.2159f),
    float3(-0.3595f,  1.3712f, -0.0114f),
    float3(-0.0447f, -0.2477f,  1.2925f)
);
static const float3x3 kAP1ToP3 = float3x3(
    float3( 2.1173f, -1.0180f, -0.0996f),
    float3(-0.2071f,  1.2150f, -0.0077f),
    float3(-0.0219f, -0.1540f,  1.1758f)
);

float3 acesRrtOdtFit(float3 v) {
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}
float3 acesFitted(float3 c2020, float3x3 outputMat) {
    float3 ap1 = mul(kRec2020ToAP1, c2020);
    ap1 = max(ap1, 0.0f.xxx);
    ap1 = acesRrtOdtFit(ap1);
    return max(mul(outputMat, ap1), 0.0f.xxx);
}

float3 hablePartial(float3 x) {
    const float A=0.15f,B=0.50f,C=0.10f,D=0.20f,E=0.02f,F=0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
float3 hableFilmic(float3 x) {
    const float3 white = hablePartial(float3(11.2f,11.2f,11.2f));
    return hablePartial(2.0f*x)/white;
}

float3 reinhardLuminance(float3 x) {
    const float lum = dot(x, float3(0.2627f,0.6780f,0.0593f));
    if (lum <= 0.0f) return 0.0f.xxx;
    const float lumTm = lum/(1.0f+lum);
    return x*(lumTm/lum);
}

static const float3x3 kAgXInputMatrix = float3x3(
    float3(0.842479f, 0.042328f, 0.042376f),
    float3(0.078434f, 0.878469f, 0.078434f),
    float3(0.079224f, 0.079166f, 0.879143f)
);
static const float3x3 kAgXOutputMatrix = float3x3(
    float3( 1.196879f, -0.052897f, -0.052972f),
    float3(-0.098021f,  1.151903f, -0.098043f),
    float3(-0.099030f, -0.098961f,  1.151074f)
);
float agxCurve(float x) {
    x = saturate(x);
    const float x2=x*x, x3=x2*x, x4=x2*x2, x5=x3*x2, x6=x3*x3;
    return 15.5f*x6-40.14f*x5+31.96f*x4-6.868f*x3+0.4298f*x2+0.1191f*x-0.00232f;
}
float3 agxCurve3(float3 v) { return float3(agxCurve(v.x),agxCurve(v.y),agxCurve(v.z)); }
float3 agxFitted(float3 c2020) {
    float3 agx = max(rec2020ToRec709(c2020), 0.0f.xxx);
    agx = mul(kAgXInputMatrix, agx);
    agx = (log2(max(agx, 2.2e-7f.xxx)) + 10.0f) / 16.5f;
    agx = agxCurve3(agx);
    return mul(kAgXOutputMatrix, agx);
}

float3 dispatchTonemapper(float3 hdr, uint tm) {
    switch (tm) {
        case 1:  return agxFitted(hdr);
        case 2:  return saturate(reinhardLuminance(rec2020ToRec709(hdr)));
        case 3:  return saturate(hableFilmic(rec2020ToRec709(hdr)));
        default: return acesFitted(hdr, kAP1ToRec709);
    }
}

float pqOetf(float L) {
    const float m1=0.1593017578125f, m2=78.84375f, c1=0.8359375f, c2=18.8515625f, c3=18.6875f;
    float Yn=saturate(L/10000.0f), Ym1=pow(Yn,m1);
    return pow((c1+c2*Ym1)/(1.0f+c3*Ym1),m2);
}
float3 pqOetf3(float3 c){return float3(pqOetf(c.x),pqOetf(c.y),pqOetf(c.z));}

float hlgOetf(float E) {
    const float a=0.17883277f, b=0.28466892f, c_hlg=0.55991073f;
    E=max(0.0f,E);
    return (E<=(1.0f/12.0f)) ? sqrt(3.0f*E) : (a*log(12.0f*E-b)+c_hlg);
}
float3 hlgOetf3(float3 c){return float3(hlgOetf(c.x),hlgOetf(c.y),hlgOetf(c.z));}

float srgbOetf(float x) {
    x=max(0.0f,x);
    return (x<=0.0031308f) ? (12.92f*x) : (1.055f*pow(x,1.0f/2.4f)-0.055f);
}
float3 srgbOetf3(float3 c){return float3(srgbOetf(c.x),srgbOetf(c.y),srgbOetf(c.z));}
float3 p3Oetf3(float3 c){return srgbOetf3(c);}

static const float kHDR10PaperWhiteNits=203.0f, kHDR10KneeNits=600.0f, kHDR10PeakNits=1000.0f;
float3 hdr10SoftClip(float3 nitsLinear) {
    const float lum=dot(nitsLinear,float3(0.2627f,0.6780f,0.0593f));
    if(lum<=0.0f) return 0.0f.xxx;
    float lumOut;
    if(lum<=kHDR10KneeNits) { lumOut=lum; }
    else {
        float t=(lum-kHDR10KneeNits)/(kHDR10PeakNits-kHDR10KneeNits);
        lumOut=kHDR10KneeNits+(kHDR10PeakNits-kHDR10KneeNits)*t/(1.0f+t);
    }
    return nitsLinear*(lumOut/lum);
}

[shader("fragment")]
float4 tonemapFrag(float4 svPos : SV_Position) : SV_Target0 {
    int2 coord=int2(svPos.xy);
    float3 hdr=max(hdrTarget[coord].xyz, 0.0f.xxx);
    float3 out3;
    switch (pushConst.outputColorSpace) {
        case 0: { float3 nits=hdr*kHDR10PaperWhiteNits; nits=hdr10SoftClip(nits); out3=pqOetf3(nits); break; }
        case 1: { out3=saturate(hlgOetf3(hdr)); break; }
        case 2: { out3=rec2020ToRec709(hdr); break; }
        case 3: {
            float3 p3 = (pushConst.tonemapper==0u) ? acesFitted(hdr,kAP1ToP3)
                                                   : mul(kRec709ToP3,dispatchTonemapper(hdr,pushConst.tonemapper));
            out3=p3Oetf3(saturate(p3)); break;
        }
        case 4: { out3=saturate(rec2020ToRec709(hdr)); break; }
        default: { out3=srgbOetf3(saturate(dispatchTonemapper(hdr,pushConst.tonemapper))); break; }
    }
    return float4(out3, 1.0f);
}
