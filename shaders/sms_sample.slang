// Native Theia specialization of the shared supported-delta sampler. Work is
// produced separately from the adaptive/replayed ordinary-PT candidate pool.
import common;
import math;
import rng;
import bsdf;
import medium_walk;
import surface;
import sms;
import sms_transport;
import sms_work;
import sms_scene;
import sms_prior;
import sms_context;

#define SMS_SCENE TheiaSmsScene
#define SMS_HIT TheiaSmsHit
#define SMS_TEXTURE_SAMPLER TheiaSmsTextureSampler
#define SMS_MATERIALS t_smsMaterials
#define SMS_EMITTERS t_smsEmitters
#define SMS_EMITTER_CDF t_smsEmitterCdf
#define SMS_STORE_LOCAL_TARGET(slot,work,value) \
    { float smsPhi=dot((value)*asfloat((work).words[5].w),float3(.2126,.7152,.0722)); \
      if (!isfinite(smsPhi) || smsPhi<0.0) { smsQueueError(pc,kSmsQueueNumericError); return; } \
      t_smsProducer[0].prefixes[slot].w=smsPhi; }
#include "sms_compute.slang"
