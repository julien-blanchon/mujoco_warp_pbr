import optix, wpx

ctx = wpx.create_context()
for n in [
  "MAX_TRACE_DEPTH",
  "MAX_TRAVERSABLE_GRAPH_DEPTH",
  "MAX_PRIMITIVES_PER_GAS",
  "MAX_INSTANCES_PER_IAS",
  "RTCORE_VERSION",
  "MAX_INSTANCE_ID",
  "NUM_BITS_INSTANCE_VISIBILITY_MASK",
  "MAX_SBT_RECORDS_PER_GAS",
  "MAX_SBT_OFFSET",
]:
  for pre in ["DEVICE_PROPERTY_LIMIT_", "DEVICE_PROPERTY_"]:
    if hasattr(optix, pre + n):
      print(n, ctx.getProperty(getattr(optix, pre + n)))
      break
