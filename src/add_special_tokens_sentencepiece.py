from sentencepiece import sentencepiece_model_pb2 as model
m = model.ModelProto()
m.ParseFromString(open('/mnt/nvme/home/shared_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3/tokenizer.model.v3', 'rb').read())

new_sys_begin = m.SentencePiece()
new_sys_begin.piece = "[SYS_INST]"
new_sys_begin.score = 0
new_sys_begin.type = 3
new_sys_end = m.SentencePiece()
new_sys_end.piece = "[/SYS_INST]"
new_sys_end.score = 0
new_sys_end.type = 3
del m.pieces[10:12]
m.pieces.insert(10, new_sys_begin)
m.pieces.insert(11, new_sys_end)

m.trainer_spec.control_symbols[7] = "[SYS_INST]"
m.trainer_spec.control_symbols[8] = "[/SYS_INST]"

with open('new.model', 'wb') as f:
    f.write(m.SerializeToString())
