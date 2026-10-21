"""Unified text + motion vocabulary (single-layer MotionGPT VQ-VAE)."""
PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 101, 102, 103
BASE_VOCAB_SIZE = 30522
SOT_ID, EOT_ID, T2M_ID, M2T_ID, M2M_ID = range(30522, 30527)
MOTION_CODE_START = 30527
CODEBOOK_SIZE = 512
TOTAL_VOCAB = MOTION_CODE_START + CODEBOOK_SIZE
FRAME_SIZE = 1
TEMPORAL_DOWNSAMPLE = 4
MAX_MOTION_FRAMES = 196
MIN_T_PRIME = 2
MAX_T_PRIME = MAX_MOTION_FRAMES // TEMPORAL_DOWNSAMPLE
MAX_TEXT_CONTENT = 50


def is_motion_id(token_id):
    return MOTION_CODE_START <= token_id < TOTAL_VOCAB


def setup_tokenizer(tokenizer):
    global PAD_ID, CLS_ID, SEP_ID, MASK_ID, BASE_VOCAB_SIZE
    global SOT_ID, EOT_ID, T2M_ID, M2T_ID, M2M_ID, MOTION_CODE_START, TOTAL_VOCAB
    special = ['<|sot|>', '<|eot|>', '<|t2m|>', '<|m2t|>', '<|m2m|>']
    motion = [f'<|motion_{i}|>' for i in range(CODEBOOK_SIZE)]
    if tokenizer.convert_tokens_to_ids(special[0]) == tokenizer.unk_token_id:
        tokenizer.add_tokens(special + motion)
    PAD_ID, CLS_ID, SEP_ID, MASK_ID = (tokenizer.pad_token_id, tokenizer.cls_token_id,
                                     tokenizer.sep_token_id, tokenizer.mask_token_id)
    SOT_ID, EOT_ID, T2M_ID, M2T_ID, M2M_ID = tokenizer.convert_tokens_to_ids(special)
    BASE_VOCAB_SIZE = SOT_ID
    MOTION_CODE_START = M2M_ID + 1
    TOTAL_VOCAB = MOTION_CODE_START + CODEBOOK_SIZE
    if tokenizer.convert_tokens_to_ids(special + motion) != list(range(SOT_ID, TOTAL_VOCAB)):
        raise ValueError('Checkpoint tokenizer has an incompatible motion vocabulary.')
    if len(tokenizer) != TOTAL_VOCAB:
        raise ValueError('Unexpected tokenizer vocabulary size.')


VQ_MODE = 'frame'

def set_vq_mode(mode):
    if mode != 'frame':
        raise ValueError('The restored release evaluator supports frame VQ checkpoints only.')
