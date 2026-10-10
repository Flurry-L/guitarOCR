"""Small shared CPU runtime for the layout and signature networks."""

import os


def cpu_session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
