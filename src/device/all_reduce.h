/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include <vector>

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nranks * chunkCount;
    ssize_t offset;
    int nelem;
    int chunk;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;

      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks - 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directSend(offset, offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvCopyDirectSend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);

      prims.directRecv(offset, offset, nelem);
    }
  }

  // rabenseifner algorithm
  // precondition: nranks is a power of 2
  // precondition: for each loop message size is divisible by nranks
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRabenseifner(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int rank = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    const int log2nranks = __log2f(nranks);
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = log2nranks * chunkCount;
    int this_nelem, target_nelem;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    // initialize log2nranks primitives and put them in a vector
    using prim_t = Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>;
    prim_t* prim_v[8];
    //std::vector<std::reference_wrapper<prim_t>> prim_v;
    int mask = 1;
    for (int i=0; i<log2nranks; i++) {
      int targetRank = rank ^ mask;
      // create primitive for target rank and put it in vector
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
        (tid, nthreads, &ring->userRanks[targetRank], &ring->userRanks[targetRank], work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      prim_v[i] = &prims;
      mask <<= 1;
    }
    //Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      //(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t this_chunkOffset, target_chunkOffset, this_offset, target_offset;

      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      int mask, targetRank, xchg_nchunks, this_chunk, target_chunk, tmp_chunk;
      // step 0: push data to next GPU
      //mask = 1;
      //targetRank = rank ^ mask;
      //xchg_nchunks = nranks/2;
      // the chunk receieved
      //this_chunk = rank & mask;
      // the chunk send to remote rank
      //target_chunk = targetRank & (mask*2-1);

      //this_chunkOffset = this_chunk * chunkCount;
      //target_chunkOffset = target_chunk * chunkCount;
      //this_offset = gridOffset + elemOffset + this_chunkOffset;
      //target_offset = gridOffset + elemOffset + target_chunkOffset;
      //nelem = chunkCount*xchg_nchunks;
      //prims.directSend(offset, offset, nelem);

      // rank:                0      1      2       3       4       5       6       7
      // binary:              000    001    010     011     100     101     110     111
      // reverse binary:      000    100    010     110     001     101     011     111
      // REDUCE SCATTER PHASE
      // step 0, xchg between 0-1 2-3 4-5 6-7
      //   mask: 1
      //   2*mask-1 = 1, reverse binary = 100
      //   xchg_nchunks: 4
      //   targetRank:        1      0      3       2       5       4       7       6
      //   this_chunk:        0      4      0       4       0       4       0       4
      //   binary:            000    100    000     100     000     100     000     100
      //   target_chunk:      4      0      4       0       4       0       4       0
      // step 1, xchg between 0-2 1-3 4-6 5-7
      //   mask: 2
      //   2*mask-1 = 3, reverse binary = 110
      //   xchg_nchunks: 2
      //   targetRank:        2      3      0       1       6       7       4       5
      //   this_chunk:        0      4      2       6       0       4       2       6
      //   binary:            000    100    010     110     000     100     010     110
      //   target_chunk:      2      6      0       4       2       6       0       4
      // step 2, xchg between 0-4 1-5 2-6 3-7
      //   mask: 4
      //   2*mask-1 = 7, reverse binary = 111
      //   xchg_nchunks: 1
      //   targetRank:        4      5      6       7       0       1       2       3
      //   this_chunk:        0      4      2       6       1       5       3       7
      //   binary:            000    100    010     110     001     101     011     111
      //   target_chunk:      1      5      3       7       0       4       2       6
      //
      // in REDUCE SCATTER phase, this_chunk = reverse binary of rank & rever binary of 2*mask-1
      //
      // ALLGATHER PHASE
      // step 3, xchg between 0-4 1-5 2-6 3-7
      //   mask: 4
      //   2*mask-1 = 7, reverse binary = 111
      //   xchg_nchunks: 1
      //   targetRank:        4      5      6       7       0       1       2       3
      //   this_chunk:        1      5      3       7       0       4       2       6
      //   binary:            001    101    011     111     000     100     010     110
      //   target_chunk:      0      4      2       6       1       5       3       7
      // step 4, xchg between 0-2 1-3 4-6 5-7
      //   mask: 2
      //   2*mask-1 = 3, reverse binary = 110
      //   xchg_nchunks: 2
      //   targetRank:        2      3      0       1       6       7       4       5
      //   this_chunk:        2      6      0       4       2       6       0       4
      //   binary:            010    110    000     100     010     110     000     100
      //   target_chunk:      0      4      2       6       1       5       3       7
      // step 5, xchg between 0-1 2-3 4-5 6-7
      //   mask: 1
      //   2*mask-1 = 1, reverse binary = 100
      //   xchg_nchunks: 4
      //   targetRank:        1      0      3       2       5       4       7       6
      //   this_chunk:        4      0      4       0       4       0       4       0
      //   binary:            100    000    100     000     100     000     100     000
      //   target_chunk:      0      4      0       4       0       4       0       4

      // log2(k) steps: reduce and copy to pair
      // setup condition for step 0
      mask = 1;
      xchg_nchunks = nranks/2;
      this_chunk = 0;
      for (int j = 0; j < log2nranks; ++j) {
        targetRank = rank ^ mask;
        if ((rank & mask) != 0) {  // targetRank, < rank
            target_chunk = this_chunk;
            this_chunk += xchg_nchunks;
        } else { // rank < targetRank
            target_chunk = this_chunk + xchg_nchunks;
        }

        this_chunkOffset = this_chunk * chunkCount;
        target_chunkOffset = target_chunk * chunkCount;
        this_offset = gridOffset + elemOffset + this_chunkOffset;
        target_offset = gridOffset + elemOffset + target_chunkOffset;
        this_nelem = (int)min(chunkCount*xchg_nchunks, remCount - this_chunkOffset);
        target_nelem = (int)min(chunkCount*xchg_nchunks, remCount - target_chunkOffset);
        prim_v[j]->directSend(target_offset, target_offset, target_nelem);
        prim_v[j]->directRecvReduceCopy(this_offset, this_offset, this_nelem);

        mask <<= 1;
        xchg_nchunks /= 2;
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      //targetRank = rank ^ mask;
      //mask >>= 1;
      //assert (xchg_nchunks == 1);
      //chunk = rank & (mask*2-1);
      //
      //chunkOffset = chunk * chunkCount;
      //offset = gridOffset + elemOffset + chunkOffset;
      //nelem = (int)min(chunkCount*xchg_nchunks, remCount - chunkOffset);
      //prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-1 steps: copy to next GPU
      xchg_nchunks = 1;
      mask >>= 1;
      // exchange target chunk and this chunk in allgather phase
      tmp_chunk = target_chunk;
      target_chunk = this_chunk;
      this_chunk = tmp_chunk;

      for (int j = log2nranks-1; j >= 0; --j) {
        targetRank = rank ^ mask;
        if ((rank & mask) != 0) { // targetRank < rank
            this_chunk = target_chunk - xchg_nchunks;
        } else { // rank < targetRank
            this_chunk = target_chunk + xchg_nchunks;
        }


        this_chunkOffset = this_chunk * chunkCount;
        target_chunkOffset = target_chunk * chunkCount;
        this_offset = gridOffset + elemOffset + this_chunkOffset;
        target_offset = gridOffset + elemOffset + target_chunkOffset;
        //nelem = (int)min(chunkCount*xchg_nchunks, remCount - chunkOffset);
        //nelem = chunkCount*xchg_nchunks;
        this_nelem = (int)min(chunkCount*xchg_nchunks, remCount - this_chunkOffset);
        target_nelem = (int)min(chunkCount*xchg_nchunks, remCount - target_chunkOffset);
        prim_v[j]->directSend(target_offset, target_offset, target_nelem);
        prim_v[j]->directRecvCopy(this_offset, this_offset, this_nelem);
        //prims.directRecvCopyDirectSend(offset, nelem);

        if ((rank & mask) != 0) // targetRank < rank
            target_chunk -= xchg_nchunks;
        xchg_nchunks *= 2;
        mask >>= 1;
      }

      // Make final copy from buffer to dest.
      //chunk = modRanks(rank + 1);
      //chunkOffset = chunk * chunkCount;
      //offset = gridOffset + elemOffset + chunkOffset;
      //nelem = (int)min(chunkCount, remCount - chunkOffset);

      //prims.directRecv(offset, offset, nelem);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 2, max number of send is 2
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<n, 1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<1, n>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
            work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    //runRing<T, RedOp, Proto>(tid, nthreads, work);
    runRabenseifner<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RABENSEIFNER, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRabenseifner<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
      runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #else
      runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #endif
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    const ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t size = work->collnet.count;
    const ssize_t loopSize = nChannels*direct->nHeads*chunkSize;

    const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0);
    const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast   = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS);
    const int nThreadsReduce = work->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;

    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, work->sendbuff, work->recvbuff,
           work->redOpArg, 2*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*direct->nHeads*chunkSize;
        int nelem = min(direct->nHeads*chunkSize, size-offset);
        if (work->regUsed) {
          prims.directScatter(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
        } else {
          prims.scatter(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
        }
      }
      // Coverity complains about a possible overrun inside the destructor of "prims", but that's actually
      // a false positive.
      // coverity[overrun-call:FALSE]
    } else if (tid >= tidStartReduce && direct->out != -1) {
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (work->regUsed) {
            prims.directRecvReduceSend(offset, nelem);
          } else {
            prims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Directly send to network
        if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
          if (tid == tidStartReduce) {
            int steps = (int)divUp(size * sizeof(T), NCCL_MAX_COLLNET_SIZE);
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, steps);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
            int nelem = min(chunkSize, size-offset);
            prims.send(offset, nelem);
          }
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, nThreadsGather, direct->up, NULL, work->sendbuff, work->recvbuff,
           work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*direct->nHeads*chunkSize;
        int nelem = min(direct->nHeads*chunkSize, size-offset);
        prims.directGather(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      if (hasDn) {
        // Recv from network, broadcast
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, work->sendbuff, work->recvbuff,
             work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopyDirectSend(offset, nelem, /*postOp=*/true);
        }
      } else {
        if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
          if (tid == tidStartBcast) {
            int steps = (int)divUp(size * sizeof(T), NCCL_MAX_COLLNET_SIZE);
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, steps);
          }
          __syncwarp();
        } else {
          // Recv from network (no post thread needed)
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid - tidStartBcast, nThreadsBcast, &direct->out, nullptr, work->sendbuff, work->recvbuff,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.recv(offset, nelem, /*postOp=*/true);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const bool hasOut = nvls->out != -1;
    const int nranks = ncclShmem.comm.nRanks;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasOut ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (work->oneNode) {
      ssize_t gridOffset, channelCount, chunkSize;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkSize);
      const ssize_t loopCount = nvls->nHeads * chunkSize;
      ssize_t offset;
      int nelem;
      int remCount = channelCount%(nvls->nHeads*chunkSize);
      int lastChunkSize = alignUp(divUp(remCount, nvls->nHeads), 16384/sizeof(T));

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          offset = gridOffset + elemOffset;
          nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          offset = gridOffset + elemOffset;
          nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce) {
        // Reduce, broadcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          chunkOffset = elemOffset + nvls->headRank * chunkSize;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkSize, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else {
      const int bid = ncclShmem.channelId - work->channelLo;
      const int nChannels = work->channelHi - work->channelLo + 1;
      const ssize_t chunkSize = work->collnet.chunkCount;
      const ssize_t loopSize = nChannels * nvls->nHeads * chunkSize;
      const ssize_t size = work->collnet.count;

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 :min(nvls->nHeads * chunkSize, size - offset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        if (!hasOut) {
          // Reduce, broadcast through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
          // Coverity complains about a possible overrun inside the class below, but that's actually
          // a false positive.
          // coverity[identity_transfer:FALSE]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
              work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.directRecvDirectSend(offset, offset, nelem);
          }
        } else {
          // Reduce, send to network
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          // Coverity complains about a possible overrun inside the class below, but that's actually
          // a false positive.
          // coverity[identity_transfer:FALSE]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->out, NULL, NULL,
              work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.directRecvDirectSend(offset, offset, nelem);
          }
        }
      } else if (tid < tidEndBcast && nvls->headRank != -1) {
        // Recv from network, broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndReduce, nThreadsBcast, &nvls->out, &nvls->down, NULL, NULL,
            work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const int treeUp = nvls->treeUp;
    const int* treeDown = nvls->treeDown;
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nvls->nHeads * chunkCount;
    const int nranks = ncclShmem.comm.nRanks;
    const bool hasUp = treeUp != -1;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasUp ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 4) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasUp ? 5 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
    ssize_t offset;
    int nelem;
    int remCount = channelCount%(nvls->nHeads*chunkCount);
    int lastChunkCount = alignUp(divUp(remCount, nvls->nHeads), 16/sizeof(T));

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (tid < tidEndScatter) {
      // Scatter
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
          work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.scatter(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // Gather
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
          work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.gather(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndReduce && nvls->headRank != -1) {
      if (!hasUp) {
        // Reduce and Broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<3>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, treeDown, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
        // FanAsymmetric<3, 1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanAsymmetric<3, 1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, &treeUp, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else if (tid < tidEndBcast && nvls->headRank != -1) {
      // Recv from network, broadcast
      using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
      // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
      // FanAsymmetric<1, 3>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, 3>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndReduce, nThreadsBcast, &treeUp, treeDown, NULL, NULL,
          work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t chunkOffset;
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        chunkOffset = elemOffset + nvls->headRank * chunkCount;
        offset = gridOffset + chunkOffset;
        nelem = min(chunkCount, channelCount - chunkOffset);
        prims.directRecvDirectSend(offset, offset, nelem);
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    ncclTree *tree = &ncclShmem.channel.collnetChain;
    ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t loopSize = int(nChannels*chunkSize);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t size = work->collnet.count;

    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, connIndex, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      // Reduce up the chain
      group = 0;
      connIndex = 1;
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      // Broadcast down the chain
      group = 1;
      connIndex = 0;
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads-nthreadsSplit;
    }

    if (tid < nthreadsSplit) {
      if (recv == -1) {
        if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
          if (groupTid == 0) {
            int steps = (int)divUp(size * sizeof(T), NCCL_MAX_COLLNET_SIZE);
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::sendPeerNotify(send, connIndex, steps);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directSend(offset, offset, nelem);
          }
        }
      } else {
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * int(chunkSize);
          int nelem = min(chunkSize, size - offset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    else {
      if (recv == nranks) {
        // I'm the first in the broadcast chain, I need to perform the division (postOp)
        if (send == -1) {
          if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
            if (groupTid == 0) {
              int steps = (int)divUp(size * sizeof(T), NCCL_MAX_COLLNET_SIZE);
              Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::recvPeerNotify(recv, connIndex, steps);
            }
            __syncwarp();
          } else {
            // Coverity reports that the callee treats &send as an array.  However, due to the use of
            // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
            // coverity[callee_ptr_arith:FALSE]
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
              prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
                work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
            for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
              ssize_t offset = gridOffset + bid * int(chunkSize);
              int nelem = min(chunkSize, size - offset);
              prims.directRecv(offset, offset, nelem, /*postOp*/true);
            }
          }
        } else {
          // Coverity reports that the callee treats &send as an array.  However, due to the use of
          // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
          // coverity[callee_ptr_arith:FALSE]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecvCopyDirectSend(offset, nelem, /*postOp*/true);
          }
        }
      } else {
        // Coverity reports that the callee treats &send as an array.  However, due to the use of
        // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        if (send == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecv(offset, offset, nelem);
          }
        } else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecvCopyDirectSend(offset, nelem);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
