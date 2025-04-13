from tinygrad import Tensor, dtypes, TinyJit

IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
MSG_PERM = Tensor([2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8], dtype=dtypes.uint32)
CHUNK_START, CHUNK_END, NODE_PARENT, NODE_ROOT = 0x01, 0x02, 0x04, 0x08
COLS = [(0,4,8,12,0,1), (1,5,9,13,2,3), (2,6,10,14,4,5), (3,7,11,15,6,7)]
DIAGS = [(0,5,10,15,8,9), (1,6,11,12,10,11), (2,7,8,13,12,13), (3,4,9,14,14,15)]

def rotr(x:Tensor, bits:int) -> Tensor: return (x.rshift(bits))|(x.lshift(x.dtype.itemsize*8-bits))

def g(state:Tensor, a:int, b:int, c:int, d:int, x:Tensor, y:Tensor) -> Tensor:
  for z, r1, r2 in [(x, 16, 12), (y, 8, 7)]:
    state[:,a] = state[:,a] + state[:,b] + z; state[:,d] = rotr(state[:,d]^state[:,a], r1);
    state[:,c] = state[:,c] + state[:,d]; state[:,b] = rotr(state[:,b]^state[:,c], r2)
  return state

def compress(block:Tensor, chain_val:Tensor, counter_low:Tensor, counter_high:Tensor, block_len:Tensor, flags:Tensor) -> Tensor:
  state = Tensor.cat(chain_val, IV[:4].expand(block.shape[0],4), Tensor.stack(counter_low, counter_high, block_len, flags, dim=1), dim=1).cast(dtypes.uint32)
  for _ in range(7):
    for a,b,c,d,xi,yi in [*COLS, *DIAGS]: state = g(state, a, b, c, d, block[:,xi], block[:,yi])
    block = block[:,MSG_PERM]
  state[:,:8] = state[:,:8] ^ state[:,8:16]; state[:,8:16] = state[:,8:16] ^ chain_val
  return state

def pack(blocks:Tensor) -> Tensor:
  block_words = Tensor.zeros(blocks.shape[0], 16, dtype=dtypes.uint32).contiguous()
  for j in range(64): k=j//4; block_words[:,k] = block_words[:,k] | blocks[:,j].lshift(j%4*8)
  return block_words

def chunk(chunk:Tensor, chain_val:Tensor, chunk_idx:Tensor, flags:Tensor, root:bool=False) -> Tensor:
  pad = (64-chunk.shape[1]%64)%64
  blocks = pack(chunk.pad((0,pad)).reshape(-1,64)).reshape(bsize:=chunk.shape[0],-1,16)
  for i in range(blocks.shape[1]):
    block_flags = flags | (NODE_ROOT if root and i==blocks.shape[1]-1 else 0) | (CHUNK_START if i==0 else 0) | (CHUNK_END if i==blocks.shape[1]-1 else 0)
    block_len = Tensor.full((bsize,), min(64, chunk.shape[1]-i*64), dtype=dtypes.uint32)
    chain_val = compress(blocks[:,i], chain_val, chunk_idx, Tensor.zeros(bsize, dtype=dtypes.uint32), block_len, block_flags)[:,:8]
  return chain_val

def parent(left:Tensor, right:Tensor, flags:Tensor) -> Tensor:
  return compress(Tensor.cat(left, right, dim=1), IV.expand(bsize:=left.shape[0],8), Tensor.zeros(bsize, dtype=dtypes.uint32),
                  Tensor.zeros(bsize, dtype=dtypes.uint32),Tensor.full((bsize,), 64, dtype=dtypes.uint32), flags|NODE_PARENT)[:,:8]

def blake3(msg:Tensor, max_batch_size:int=None) -> Tensor:
  if not isinstance(msg, Tensor): msg = Tensor(msg, dtype=dtypes.uint8).flatten()
  msg = msg.cast(dtypes.uint8)
  if msg.shape[0] == 0:
    return compress(Tensor.zeros((1, 16), dtype=dtypes.uint32), IV.expand(1, 8), Tensor.zeros(1, dtype=dtypes.uint32), Tensor.zeros(1, dtype=dtypes.uint32),
                    Tensor.zeros(1, dtype=dtypes.uint32), Tensor.full((1,), CHUNK_START | CHUNK_END | NODE_ROOT, dtype=dtypes.uint32))[:,:8][0]
  num_chunks = (msg.shape[0] + 1023) // 1024
  chunks = msg.pad((0, num_chunks * 1024 - msg.shape[0]), value=0).reshape(num_chunks, 1024)
  chunk_hashes = Tensor.zeros((num_chunks, 8), dtype=dtypes.uint32).contiguous()
  chunk_indices = Tensor.arange(num_chunks, dtype=dtypes.uint32)
  last_chunk_complete = (msg.shape[0] % 1024 == 0)
  last_chunk_idx = num_chunks - 1
  batch_size = (num_chunks if last_chunk_complete and num_chunks > 1 else last_chunk_idx) if max_batch_size is None \
    else min(max_batch_size, num_chunks if last_chunk_complete and num_chunks > 1 else last_chunk_idx)

  for i in range(0, last_chunk_idx if batch_size else 0, max(batch_size, 1)):
    batch_end = min(i + batch_size, last_chunk_idx  + (1 if last_chunk_complete else 0))
    batch_chunks = chunks[i:batch_end]
    batch_indices = chunk_indices[i:batch_end]
    chunk_hashes[i:batch_end] = chunk(batch_chunks, IV.expand(batch_end - i, 8), batch_indices, Tensor.zeros((batch_end - i,), dtype=dtypes.uint32))
  if (not last_chunk_complete and num_chunks > 0) or num_chunks == 1:
    last_chunk_size = msg.shape[0] % 1024 if not last_chunk_complete else 1024
    last_chunk = chunks[last_chunk_idx:last_chunk_idx+1, :last_chunk_size]
    chunk_hashes[last_chunk_idx:last_chunk_idx+1] = chunk(last_chunk, IV.expand(1, 8), chunk_indices[last_chunk_idx:last_chunk_idx+1], Tensor.zeros((1,), dtype=dtypes.uint32), num_chunks==1)
  while chunk_hashes.shape[0] > 1:
    num_parents = (chunk_hashes.shape[0] + 1) // 2
    parents = Tensor.zeros((num_parents, 8), dtype=dtypes.uint32).contiguous()
    pair_count = chunk_hashes.shape[0] // 2
    batch_left = chunk_hashes[0:-1:2] if chunk_hashes.shape[0] % 2 else chunk_hashes[::2]
    batch_right = chunk_hashes[1::2]
    parent_flags = Tensor.full((pair_count,), NODE_ROOT if num_parents == 1 else 0, dtype=dtypes.uint32)
    parents[:pair_count] = parent(batch_left, batch_right, parent_flags)
    if chunk_hashes.shape[0] % 2: parents[-1] = chunk_hashes[-1]
    chunk_hashes = parents
  return chunk_hashes[0]
