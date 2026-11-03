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
import sms_resampling;

#define SMS_SCENE TheiaSmsScene
#define SMS_HIT TheiaSmsHit
#define SMS_TEXTURE_SAMPLER TheiaSmsTextureSampler
#define SMS_MATERIALS t_smsMaterials
#define SMS_EMITTERS t_smsEmitters
#define SMS_EMITTER_CDF t_smsEmitterCdf
#define SMS_STORE_LOCAL_TARGET(slot,work,value) \
    { bool smsReuse=(t_smsProducer[0].flags&1u)!=0u; \
      float smsPhi=t_smsProducer[0].prefixes[slot].w; \
      if (!smsCacheLocalTarget(smsReuse,(value),asfloat((work).words[5].w),smsPhi)) \
          { smsQueueError(pc,kSmsQueueNumericError); return; } \
      if (smsReuse) t_smsProducer[0].prefixes[slot].w=smsPhi; }
#include "sms_compute.slang"
