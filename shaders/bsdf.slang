import common;
import math;
__exported import bsdf_shared;

float3 evalBSDF(GpuMaterial mat, float3 wo, float3 wi, float3 N, float3 T, float3 B, float3 cN, float3 cT, float3 cB)
{
    float3 Ns;
    float3 Ts;
    float3 Bs;
    orientFrame(wo, N, T, B, Ns, Ts, Bs);
    float3 woL = toLocal(wo, Ts, Bs, Ns);
    float3 wiL = toLocal(wi, Ts, Bs, Ns);
    if (woL.z == 0.0 || wiL.z == 0.0) {
        return 0.0.xxx;
    }

    float3 baseColor = saturate3(mat.baseColorWeight.xyz);
    float3 specColor = saturate3(mat.specularColorWeight.xyz);
    float3 transColor = effectiveTransColor(mat);
    float3 subsurfaceColor = saturate3(mat.subsurfaceColorWeight.xyz);
    float3 coatColor = saturate3(mat.coatColorWeight.xyz);
    float3 fuzzColor = saturate3(mat.fuzzColorWeight.xyz);

    float baseRough = mat.specularRoughAnisoIor.x;
    float baseAniso = mat.specularRoughAnisoIor.y;
    float coatRough = mat.coatRoughAnisoIorDark.x;
    float coatAniso = mat.coatRoughAnisoIorDark.y;
    float eta = max(mat.specularRoughAnisoIor.z, 1.01);
    float coatEta = max(mat.coatRoughAnisoIorDark.z, 1.01);
    float diffuseRough = saturate(mat.baseMetalnessDiffRough.y);
    float coatDark = saturate(mat.coatRoughAnisoIorDark.w);

    float effSpecRough = openpbrEffectiveSpecularRoughness(baseRough, coatRough, saturate(mat.coatColorWeight.w));
    float alphaX;
    float alphaY;
    computeAlpha(effSpecRough, baseAniso, alphaX, alphaY);
    float coatAlphaX;
    float coatAlphaY;
    computeAlpha(coatRough, coatAniso, coatAlphaX, coatAlphaY);

    float diffuseWeight;
    float subsurfaceWeight;
    float specularWeight;
    float metalWeight;
    float transmissionWeight;
    float coatWeight;
    float fuzzWeight;
    float diffuseTransWeight;
    bool mirrorMode;
    bool glassMode;
    computeLobeWeights(mat, diffuseWeight, subsurfaceWeight, specularWeight, metalWeight, transmissionWeight, coatWeight, fuzzWeight, diffuseTransWeight, mirrorMode, glassMode);

    float ssAniso = clamp(mat.opacityFlagsPad.z, -1.0, 1.0);
    // OpenPBR coat darkening (G1) + coat_color absorption (G2) — per-channel view-independent
    // multiply, matching Hyperion/bsdf_shared. Fuzz (topmost) then attenuates the base.
    float3 Emetal = baseColor * saturate(mat.specularColorWeight.w);
    float3 Edielectric = lerp(baseColor, subsurfaceColor, saturate(mat.subsurfaceColorWeight.w));
    float3 Ebase = lerp(Edielectric, Emetal, saturate(mat.baseMetalnessDiffRough.x));
    float3 baseLayerScale = openpbrCoatDarkening(Ebase, coatEta, coatWeight, coatDark)
                          * openpbrCoatColorAttenuation(coatColor, coatWeight);
    baseLayerScale *= 1.0 - mx_zeltner_sheen_dir_albedo(woL.z, saturate(mat.fuzzRoughPad.x)) * fuzzWeight;

    float3 result = 0.0.xxx;
    if (wiL.z > 0.0 && woL.z > 0.0) {
        float3 dielectricF0 = iorToF0(eta);
        float3 glossyF0 = lerp(dielectricF0 * specColor, baseColor, saturate(mat.baseMetalnessDiffRough.x));
        float3 glossyF82 = specColor;
        // Dielectric base diffuse/subsurface sit UNDER the specular layer: attenuate by the
        // specular layer's directional albedo, symmetric in NoV/NoL so it stays reciprocal.
        float specF0 = luminance(dielectricF0 * specColor);
        float specAlphaAvg = safeSqrt(alphaX * alphaY);
        float diffuseUnderSpec = saturate(1.0 - fresnelGgxDirAlbedo(woL.z, specF0, specAlphaAvg)) *
                                 saturate(1.0 - fresnelGgxDirAlbedo(wiL.z, specF0, specAlphaAvg));

        if (diffuseWeight > 0.0) {
            result += baseLayerScale * diffuseUnderSpec * diffuseWeight * evalDiffuse(baseColor, diffuseRough, woL, wiL);
        }
        if (subsurfaceWeight > 0.0) {
            float ssScale = max(mat.subsurfaceRadiusScale.w, 0.001);
            float3 ssTint = subsurfaceColor * (1.0 / (1.0 + ssScale * (mat.subsurfaceRadiusScale.x + mat.subsurfaceRadiusScale.y + mat.subsurfaceRadiusScale.z)));
            // Bulk subsurface: forward scattering (g>0) reduces back-reflected energy.
            // (In thin-walled mode the split is already baked into the lobe weights.)
            float ssReflScale = (mat.opacityFlagsPad.w > 0.5) ? 1.0 : saturate(1.0 - ssAniso);
            result += baseLayerScale * diffuseUnderSpec * subsurfaceWeight * ssReflScale * evalDiffuse(ssTint, saturate(0.25 + 0.5 * diffuseRough), woL, wiL);
        }
        if (specularWeight > 0.0 || metalWeight > 0.0) {
            result += baseLayerScale * (specularWeight + metalWeight) * evalReflectionMicrofacetThinFilm(dielectricF0 * specColor, baseColor, glossyF82, mat.baseMetalnessDiffRough.x, woL, wiL, alphaX, alphaY, mat.thinFilmParams.x, max(mat.thinFilmParams.y, 1.0), mat.thinFilmParams.z);
        }
        if (fuzzWeight > 0.0) {
            result += fuzzWeight * evalSheen(fuzzColor, saturate(mat.fuzzRoughPad.x), woL, wiL);
        }
    }

    // Coat reflection lobe — evaluated in its own frame so geometry_coat_normal /
    // geometry_coat_tangent can orient the clear-coat independently of the base.
    if (coatWeight > 0.0 && woL.z > 0.0 && wiL.z > 0.0) {
        float3 cNs;
        float3 cTs;
        float3 cBs;
        orientFrame(wo, cN, cT, cB, cNs, cTs, cBs);
        float3 woC = toLocal(wo, cTs, cBs, cNs);
        float3 wiC = toLocal(wi, cTs, cBs, cNs);
        if (woC.z > 0.0 && wiC.z > 0.0) {
            float3 coatF0 = iorToF0(coatEta) * max(coatColor, 0.04.xxx);
            result += coatWeight * evalReflectionMicrofacetThinFilm(coatF0, coatF0, coatColor, 0.0, woC, wiC, coatAlphaX, coatAlphaY, mat.thinFilmParams.x, max(mat.thinFilmParams.y, 1.0), mat.thinFilmParams.z);
        }
    }

    if (transmissionWeight > 0.0) {
        float transAlphaX = alphaX; // Use base layer roughness for transmission
        float transAlphaY = alphaY;
        result += transmissionWeight * evalTransmissionMicrofacet(transColor, mat.transmissionParams.x, eta, transAlphaX, transAlphaY, woL, wiL);
    }

    // Thin-walled diffuse transmission (subsurface sheet): Lambertian-ish lobe into the
    // back hemisphere, tinted by subsurface_color.
    if (diffuseTransWeight > 0.0 && woL.z > 0.0 && wiL.z < 0.0) {
        float3 wiR = float3(wiL.x, wiL.y, -wiL.z);
        result += diffuseTransWeight * evalDiffuse(subsurfaceColor, diffuseRough, woL, wiR);
    }

    return max(result, 0.0.xxx);
}

float pdfBSDF(GpuMaterial mat, float3 wo, float3 wi, float3 N, float3 T, float3 B, float3 cN, float3 cT, float3 cB)
{
    float3 Ns;
    float3 Ts;
    float3 Bs;
    orientFrame(wo, N, T, B, Ns, Ts, Bs);
    float3 woL = toLocal(wo, Ts, Bs, Ns);
    float3 wiL = toLocal(wi, Ts, Bs, Ns);
    if (woL.z == 0.0 || wiL.z == 0.0) {
        return 0.0;
    }

    float diffuseWeight;
    float subsurfaceWeight;
    float specularWeight;
    float metalWeight;
    float transmissionWeight;
    float coatWeight;
    float fuzzWeight;
    float diffuseTransWeight;
    bool mirrorMode;
    bool glassMode;
    computeLobeWeights(mat, diffuseWeight, subsurfaceWeight, specularWeight, metalWeight, transmissionWeight, coatWeight, fuzzWeight, diffuseTransWeight, mirrorMode, glassMode);

    if (mirrorMode) {
        float3 mDir = reflect(-woL, float3(0.0, 0.0, 1.0));
        return all(abs(mDir - wiL) < 1.0e-4.xxx) ? 1.0 : 0.0;
    }

    float baseRough = mat.specularRoughAnisoIor.x;
    float baseAniso = mat.specularRoughAnisoIor.y;
    float coatRough = mat.coatRoughAnisoIorDark.x;
    float coatAniso = mat.coatRoughAnisoIorDark.y;
    float effSpecRough = openpbrEffectiveSpecularRoughness(baseRough, coatRough, saturate(mat.coatColorWeight.w));
    float alphaX;
    float alphaY;
    computeAlpha(effSpecRough, baseAniso, alphaX, alphaY);
    float coatAlphaX;
    float coatAlphaY;
    computeAlpha(coatRough, coatAniso, coatAlphaX, coatAlphaY);

    float sum = diffuseWeight + subsurfaceWeight + specularWeight + metalWeight + transmissionWeight + coatWeight + fuzzWeight + diffuseTransWeight;
    if (sum <= 0.0) {
        return 0.0;
    }

    float pdf = 0.0;
    if (wiL.z > 0.0) {
        float cosinePdf = wiL.z * kInvPi;
        pdf += (diffuseWeight + subsurfaceWeight + fuzzWeight) * cosinePdf;
        pdf += (specularWeight + metalWeight) * reflectionPdf(woL, wiL, alphaX, alphaY);
    } else {
        if (transmissionWeight > 0.0) {
            float eta = mat.specularRoughAnisoIor.z;
            float eta_ratio = woL.z > 0.0 ? (1.0 / eta) : eta;
            float3 h = normalize(-woL * eta_ratio + wiL);
            if (h.z < 0.0) h = -h;
            float transAlphaX = alphaX;
            float transAlphaY = alphaY;
            pdf += transmissionWeight * transmissionPdf(eta, transAlphaX, transAlphaY, woL, wiL, h);
        }
        pdf += diffuseTransWeight * abs(wiL.z) * kInvPi;
    }

    // Coat reflection pdf in the (possibly independent) coat frame.
    if (coatWeight > 0.0 && woL.z > 0.0 && wiL.z > 0.0) {
        float3 cNs;
        float3 cTs;
        float3 cBs;
        orientFrame(wo, cN, cT, cB, cNs, cTs, cBs);
        float3 woC = toLocal(wo, cTs, cBs, cNs);
        float3 wiC = toLocal(wi, cTs, cBs, cNs);
        if (woC.z > 0.0 && wiC.z > 0.0) {
            pdf += coatWeight * reflectionPdf(woC, wiC, coatAlphaX, coatAlphaY);
        }
    }

    return pdf / sum;
}

BSDFSample sampleBSDF(GpuMaterial mat, float3 wo, float3 N, float3 T, float3 B, float3 cN, float3 cT, float3 cB, inout uint seed)
{
    float3 Ns;
    float3 Ts;
    float3 Bs;
    orientFrame(wo, N, T, B, Ns, Ts, Bs);
    float3 woL = toLocal(wo, Ts, Bs, Ns);

    BSDFSample sample;
    sample.direction = Ns;
    sample.weight = 0.0.xxx;
    sample.pdf = 0.0;
    sample.isDelta = false;
    // Step 3 (bulk subsurface random walk) is Hyperion-native for now (see
    // theia-realtime-volumetric in the roadmap); Theia keeps the existing tinted-diffuse
    // subsurface approximation and never enters the medium.
    sample.entersMedium = false;
    sample.mediumSigmaT = 0.0;
    sample.mediumAlbedo = 0.0.xxx;
    sample.mediumG = 0.0;
    sample.mediumEta = 1.0;

    if (abs(woL.z) <= 1.0e-5) {
        return sample;
    }

    float diffuseWeight;
    float subsurfaceWeight;
    float specularWeight;
    float metalWeight;
    float transmissionWeight;
    float coatWeight;
    float fuzzWeight;
    float diffuseTransWeight;
    bool mirrorMode;
    bool glassMode;
    computeLobeWeights(mat, diffuseWeight, subsurfaceWeight, specularWeight, metalWeight, transmissionWeight, coatWeight, fuzzWeight, diffuseTransWeight, mirrorMode, glassMode);

    float eta = max(mat.specularRoughAnisoIor.z, 1.01);
    float baseRough = mat.specularRoughAnisoIor.x;
    float baseAniso = mat.specularRoughAnisoIor.y;
    float coatRough = mat.coatRoughAnisoIorDark.x;
    float coatAniso = mat.coatRoughAnisoIorDark.y;
    float effSpecRough = openpbrEffectiveSpecularRoughness(baseRough, coatRough, saturate(mat.coatColorWeight.w));
    float alphaX;
    float alphaY;
    computeAlpha(effSpecRough, baseAniso, alphaX, alphaY);
    float coatAlphaX;
    float coatAlphaY;
    computeAlpha(coatRough, coatAniso, coatAlphaX, coatAlphaY);

    if (mirrorMode) {
        float3 wiMirror = reflect(-woL, float3(0.0, 0.0, 1.0));
        sample.direction = toWorld(wiMirror, Ts, Bs, Ns);
        sample.weight = saturate3(mat.baseColorWeight.xyz + mat.specularColorWeight.xyz);
        sample.pdf = 1.0;
        sample.isDelta = true;
        return sample;
    }

    float total = diffuseWeight + subsurfaceWeight + specularWeight + metalWeight + transmissionWeight + coatWeight + fuzzWeight + diffuseTransWeight;
    if (total <= 0.0) {
        float3 wi = sampleCosineHemisphere(randFloat2(seed));
        sample.direction = toWorld(wi, Ts, Bs, Ns);
        sample.weight = 0.0.xxx;
        sample.pdf = wi.z * kInvPi;
        return sample;
    }

    float xi = randFloat(seed) * total;
    float accum = diffuseWeight;
    uint lobe = 0u;
    if (xi >= accum) { accum += subsurfaceWeight; lobe = 1u; }
    if (xi >= accum) { accum += specularWeight + metalWeight; lobe = 2u; }
    if (xi >= accum) { accum += transmissionWeight; lobe = 3u; }
    if (xi >= accum) { accum += coatWeight; lobe = 4u; }
    if (xi >= accum) { accum += fuzzWeight; lobe = 5u; }
    if (xi >= accum) { lobe = 6u; }

    float3 wiLocal = 0.0.xxx;
    bool haveCoatDir = false;
    if (lobe == 0u || lobe == 1u || lobe == 5u) {
        wiLocal = sampleCosineHemisphere(randFloat2(seed));
    } else if (lobe == 6u) {
        // Thin-walled diffuse transmission into the back hemisphere.
        float3 wh = sampleCosineHemisphere(randFloat2(seed));
        wiLocal = float3(wh.x, wh.y, -wh.z);
    } else if (lobe == 4u) {
        // Coat reflection sampled in its own (possibly independent) frame.
        float3 cNs;
        float3 cTs;
        float3 cBs;
        orientFrame(wo, cN, cT, cB, cNs, cTs, cBs);
        float3 woC = toLocal(wo, cTs, cBs, cNs);
        if (woC.z <= 0.0) {
            return sample;
        }
        float3 m = sampleGGX_VNDF(woC, coatAlphaX, coatAlphaY, randFloat2(seed));
        float3 wiC = reflect(-woC, m);
        if (wiC.z <= 0.0) {
            return sample;
        }
        sample.direction = normalize(toWorld(wiC, cTs, cBs, cNs));
        haveCoatDir = true;
    } else if (lobe == 2u) {
        float3 m = sampleGGX_VNDF(woL, alphaX, alphaY, randFloat2(seed));
        wiLocal = reflect(-woL, m);
        if (wiLocal.z <= 0.0) {
            return sample;
        }
    } else {
        bool thinWalled = mat.opacityFlagsPad.w > 0.5;
        // Delta branches set pdf = 1 manually, bypassing pdfBSDF's /sum normalization,
        // so they must compensate for the probability of selecting this lobe.
        float invPLobe = total / max(transmissionWeight, 1.0e-6);
        float F = fresnelDielectric(abs(woL.z), eta);
        if (glassMode && (randFloat(seed) < F)) {
            // Specular Fresnel reflection chosen with probability F; the carried
            // reflectance is F, so the importance weight is F/F = 1 (energy-conserving).
            float3 wiReflect = reflect(-woL, float3(0.0, 0.0, 1.0));
            sample.direction = toWorld(wiReflect, Ts, Bs, Ns);
            sample.weight = invPLobe.xxx;
            sample.pdf = 1.0;
            sample.isDelta = true;
            return sample;
        }

        if (thinWalled) {
            // Thin sheet: light passes straight through without refractive bend.
            // Chosen with probability (1-F); the (1-F) transmittance cancels.
            float3 wiThin = -woL;
            sample.direction = normalize(toWorld(wiThin, Ts, Bs, Ns));
            float3 absorption = exp(-max(mat.transmissionParams.x, 0.001) * (1.0.xxx - effectiveTransColor(mat)));
            sample.weight = effectiveTransColor(mat) * absorption * invPLobe;
            sample.pdf = 1.0;
            sample.isDelta = true;
            return sample;
        }

        float dispScale = max(mat.transmissionParams.z, 0.0);
        if (dispScale > 0.0) {
            // Chromatic dispersion: pick a hero channel and refract with a per-channel IOR
            // derived from the Abbe number (OpenPBR transmission_dispersion_abbe_number).
            float abbe = max(mat.transmissionParams.w, 1.0);
            uint ch = min(uint(randFloat(seed) * 3.0), 2u);
            float spread = dispScale * (eta - 1.0) / abbe;
            float etaC = eta + spread * (float(ch) - 1.0);
            etaC = max(etaC, 1.001);
            float etaRatioC = woL.z > 0.0 ? (1.0 / etaC) : etaC;
            float3 m = sampleGGX_VNDF(float3(woL.x, woL.y, abs(woL.z)), alphaX, alphaY, randFloat2(seed));
            if (dot(woL, m) < 0.0) {
                m = -m;
            }
            float3 wiDisp = refract(-woL, m, etaRatioC);
            bool tir = all(abs(wiDisp) < 1.0e-6.xxx);
            if (tir) {
                wiDisp = reflect(-woL, m);
            }
            sample.direction = normalize(toWorld(wiDisp, Ts, Bs, Ns));
            float3 channelMask = (ch == 0u) ? float3(3.0, 0.0, 0.0) : ((ch == 1u) ? float3(0.0, 3.0, 0.0) : float3(0.0, 0.0, 3.0));
            float3 absorption = exp(-max(mat.transmissionParams.x, 0.001) * (1.0.xxx - effectiveTransColor(mat)));
            sample.weight = effectiveTransColor(mat) * absorption * channelMask * invPLobe;
            sample.pdf = 1.0;
            sample.isDelta = true;
            return sample;
        }

        // Smooth dielectric: perfect specular refraction is a delta BTDF. computeAlpha
        // floors roughness to alpha>=0.001, so routing roughness~0 glass through the
        // microfacet path spreads a sharp refraction into visible frost. Per OpenPBR/PBRT,
        // handle effectively-smooth roughness as an exact Snell-refraction delta lobe.
        const float kSmoothDielectricRough = 0.05;
        if (baseRough < kSmoothDielectricRough) {
            float etaRatioD = woL.z > 0.0 ? (1.0 / eta) : eta;
            float3 mD = float3(0.0, 0.0, woL.z > 0.0 ? 1.0 : -1.0);
            float3 wiRefract = refract(-woL, mD, etaRatioD);
            if (all(abs(wiRefract) < 1.0e-6.xxx)) {
                // Total internal reflection -> specular reflect (still a delta event).
                wiRefract = reflect(-woL, mD);
            }
            sample.direction = normalize(toWorld(wiRefract, Ts, Bs, Ns));
            float3 absorptionD = exp(-max(mat.transmissionParams.x, 0.001) * (1.0.xxx - effectiveTransColor(mat)));
            sample.weight = effectiveTransColor(mat) * absorptionD * invPLobe;
            sample.pdf = 1.0;
            sample.isDelta = true;
            return sample;
        }

        float etaRatio = woL.z > 0.0 ? (1.0 / eta) : eta;
        float3 m = sampleGGX_VNDF(float3(woL.x, woL.y, abs(woL.z)), alphaX, alphaY, randFloat2(seed));
        if (dot(woL, m) < 0.0) {
            m = -m;
        }
        wiLocal = refract(-woL, m, etaRatio);
        if (all(abs(wiLocal) < 1.0e-6.xxx)) {
            wiLocal = reflect(-woL, m);
            sample.isDelta = true;
        }
    }

    if (!haveCoatDir) {
        sample.direction = normalize(toWorld(wiLocal, Ts, Bs, Ns));
    }
    if (!all(isfinite(sample.direction))) {
        sample.weight = 0.0.xxx;
        sample.pdf = 0.0;
        return sample;
    }
    // A coat sampled in a tilted coat frame may land in the base back hemisphere,
    // where NEE (which rejects dot(wi,N)<=0) cannot match it — reject for MIS consistency.
    if (haveCoatDir && dot(sample.direction, Ns) <= 0.0) {
        sample.weight = 0.0.xxx;
        return sample;
    }
    sample.pdf = pdfBSDF(mat, wo, sample.direction, N, T, B, cN, cT, cB);
    if (!isfinite(sample.pdf) || sample.pdf <= 0.0) {
        sample.weight = 0.0.xxx;
        sample.pdf = 0.0;
        return sample;
    }

    // Transmission absorption is already folded into evalTransmissionMicrofacet(), so it
    // must NOT be re-applied here for the generic (non-delta) refraction path.
    float3 f = evalBSDF(mat, wo, sample.direction, N, T, B, cN, cT, cB);
    float cosTheta = abs(dot(sample.direction, Ns));
    sample.weight = f * (cosTheta / sample.pdf);
    sample.weight = all(isfinite(sample.weight)) ? max(sample.weight, 0.0.xxx) : 0.0.xxx;
    return sample;
}
