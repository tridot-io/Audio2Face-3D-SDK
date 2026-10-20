"""Fold the two constant NonZero lookups of the A2F diffusion network so the TensorRT engine can be recorded as a
CUDA graph (ProjectAvadot on-device, 2026-10-20).

Each NonZero only finds a fixed entry of the timestep table onnx::Gather_284 = [0, 999]; it depends on the batch size,
never on input values, so the result is [[1]] / [[0]] for every batch size (checked for 1..8 below). TensorRT keeps them
as NonZero layers with a device-to-host shape copy, which cannot be captured (capturing crashed the game).

usage: python fold_a2f_diffusion_nonzero.py network.onnx network_nzfold.onnx
then build with the same settings as the shipped engine, e.g.
  trtexec --onnx=network_nzfold.onnx --saveEngine=network.trt --fp16 --hardwareCompatibilityLevel=ampere+
          --minShapes=window:1x16000,identity:1x3,emotion:1x30x10,input_latents:2x2x1x256,noise:1x3x60x88831
          --optShapes=(same) --maxShapes=(same) --skipInference
Verified: onnxruntime CPU outputs of the original and folded model are bit-identical.
"""
import onnx, numpy as np, sys
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
T = inits["onnx::Gather_284"]; C0 = inits["/Constant_output_0"]
print("table onnx::Gather_284 =", T.tolist(), "dtype", T.dtype, "| /Constant_output_0 =", C0.tolist())
# Evaluate each NonZero exactly as the graph does, for batch sizes 1..8.
def eval_nz(fill):
    outs = set()
    for B in range(1, 9):
        idx = np.full((B,), fill, dtype=np.float32).astype(np.int64)   # ConstantOfShape -> Cast
        g3 = T[idx]                                                     # Gather(T, idx, axis 0)
        g1 = g3[int(C0)] if np.ndim(C0) == 0 else g3[C0]                # Gather(., const 0, axis 0)
        nz = np.array(np.nonzero(np.equal(T, g1)), dtype=np.int64)      # Equal -> NonZero
        outs.add((nz.shape, tuple(nz.ravel().tolist())))
    return outs
vals = {}
fills = {"/model/NonZero": 1.0, "/model_1/NonZero": 0.0}
# The fill values come from the ConstantOfShape nodes feeding each lookup; stop if the network differs from the one analysed.
cos = {n.name: float(numpy_helper.to_array(n.attribute[0].t).ravel()[0]) for n in g.node if n.op_type == "ConstantOfShape"}
assert cos.get("/ConstantOfShape") == 1.0 and cos.get("/ConstantOfShape_4") == 0.0, cos
assert sorted(n.name for n in g.node if n.op_type == "NonZero") == sorted(fills), "unexpected NonZero layers"
for name, fill in fills.items():
    r = eval_nz(fill); print(name, "fill", fill, "-> results over batch 1..8:", r)
    assert len(r) == 1
    vals[name] = np.array(list(r)[0][1], dtype=np.int64).reshape(list(r)[0][0])
# Replace each NonZero node by a Constant producing the same output name.
new_nodes = []
for n in g.node:
    if n.op_type == "NonZero" and n.name in vals:
        new_nodes.append(helper.make_node("Constant", [], list(n.output), name=n.name + "_folded",
                                          value=numpy_helper.from_array(vals[n.name], n.name + "_folded_value")))
        print("replaced", n.name, "with constant", vals[n.name].tolist())
    else:
        new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print("saved", dst)
