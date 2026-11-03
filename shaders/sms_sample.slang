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

#define SMS_SCENE TheiaSmsScene
#define SMS_HIT TheiaSmsHit
#define SMS_TEXTURE_SAMPLER TheiaSmsTextureSampler
#define SMS_MATERIALS t_smsMaterials
#define SMS_EMITTERS t_smsEmitters
#define SMS_EMITTER_CDF t_smsEmitterCdf
#include "sms_compute.slang"
