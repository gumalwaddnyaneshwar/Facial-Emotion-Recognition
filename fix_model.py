import h5py
import json

model_path = "deepfer_mobilenetv2_v2.h5"  # adjust if your filename differs

with h5py.File(model_path, 'r+') as f:
    raw = f.attrs.get('model_config')
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    config = json.loads(raw)

    def strip_quant(obj):
        if isinstance(obj, dict):
            obj.pop('quantization_config', None)
            for v in obj.values():
                strip_quant(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_quant(item)

    strip_quant(config)
    f.attrs.modify('model_config', json.dumps(config))

print("Done — quantization_config stripped.")