// qsa.metal's qsa_select_body<CACHE>, the selection of one query with CACHE
// consecutive blocks per thread: qsa.comp includes this once per CACHE, with
// QSA_CACHE the count, QSA_SELECT_BODY the instance's name, and
// QSA_EACH_KEY(M) and QSA_EACH_QUAD(M) expanding M(0) .. M(CACHE - 1) and
// M(0) .. M(CACHE / 4 - 1) (GLSL has no templates). With QSA_SELECT_CHUNKS
// the instance also takes rows of more than one chunk (CACHE * QSA_SELECT_TG
// blocks).
//
// A thread's keys are a register array and every walk over them is written
// out per key: an index that is not a constant would turn the array into a
// branch per element. A key past the row's end is QSA_NO_KEY, which no walk
// counts or takes, so the walks do not test the block index. A row of more
// than one chunk is walked chunk by chunk with the keys re-read into the
// same registers (Metal: the first chunk's stay, the later ones are read
// where they are used), so a walk is the same code for every chunk.
#ifndef QSA_LOAD_QUAD
// Four keys by one load where the row holds all four.
#define QSA_LOAD_QUAD(q)                                                                     \
    {                                                                                        \
        uint b = b0 + t0 + 4 * (q);                                                          \
        if (b + 4 <= nb) {                                                                   \
            uvec4 k = qsa_key(F32x4(row).v[b >> 2]);                                         \
            keys[4 * (q)] = k.x;                                                             \
            keys[4 * (q) + 1] = k.y;                                                         \
            keys[4 * (q) + 2] = k.z;                                                         \
            keys[4 * (q) + 3] = k.w;                                                         \
        } else {                                                                             \
            keys[4 * (q)] = qsa_key_at(row, b, nb);                                          \
            keys[4 * (q) + 1] = qsa_key_at(row, b + 1, nb);                                  \
            keys[4 * (q) + 2] = qsa_key_at(row, b + 2, nb);                                  \
            keys[4 * (q) + 3] = qsa_key_at(row, b + 3, nb);                                  \
        }                                                                                    \
    }
// Top digit: the lane's smallest bin from `low` up, and its keys in `leader`.
#define QSA_TOP_NEXT(i)                                                                      \
    next = min(next, (keys[i] >> 24) - low < 0x80u - low ? keys[i] >> 24 : 0xFFu);
#define QSA_TOP_COUNT(i) count += (keys[i] >> 24) == leader ? 1u : 0u;
#define QSA_COUNT(i)                                                                         \
    if ((keys[i] & mask) == prefix) atomicAdd(hist[which][(keys[i] >> shift) & 0xFFu], 1u);
#define QSA_TALLY(i)                                                                         \
    n_above += int(keys[i]) > int(threshold) ? 1u : 0u;                                      \
    n_tie += keys[i] == threshold ? 1u : 0u;
// A key above the threshold lands after the taken keys before it, a tie
// within the quota likewise: the same slot, a + min(t, quota).
#define QSA_PLACE(i)                                                                         \
    {                                                                                        \
        bool above = int(keys[i]) > int(threshold);                                          \
        bool tie = keys[i] == threshold;                                                     \
        if (above || (tie && t < quota)) sel_row.v[a + min(t, quota)] = b0 + t0 + (i);       \
        a += above ? 1u : 0u;                                                                \
        t += tie ? 1u : 0u;                                                                  \
    }
#endif

// The chunks of a row: one, whose keys the registers hold from the start,
// or (QSA_SELECT_CHUNKS) several, each re-read from the score row by every
// walk when the row has more than one.
#undef QSA_EACH_CHUNK
#undef QSA_CHUNK_KEYS
#ifdef QSA_SELECT_CHUNKS
#define QSA_EACH_CHUNK for (uint b0 = 0; b0 < nb; b0 += CHUNK)
#define QSA_CHUNK_KEYS                                                                       \
    if (nb > CHUNK) {                                                                        \
        QSA_EACH_QUAD(QSA_LOAD_QUAD)                                                         \
    }
#else
#define QSA_EACH_CHUNK
#define QSA_CHUNK_KEYS
#endif

void QSA_SELECT_BODY(F32 row, U32 sel_row, uint nb, uint k_max) {
    const uint CACHE = QSA_CACHE;
    const uint CHUNK = CACHE * QSA_SELECT_TG;
    uint tid = thread_index_in_threadgroup;
    uint t0 = tid * CACHE;
    uint keys[QSA_CACHE];
    {
        const uint b0 = 0;
        QSA_EACH_QUAD(QSA_LOAD_QUAD)
    }
    if (tid < 256) {
        hist[0][tid] = 0u;
        hist[1][tid] = 0u;
    }
    barrier();

    uint prefix = 0;
    uint mask = 0;
    uint k_rem = k_max;
    for (uint pass = 0; pass < 4; ++pass) {
        uint shift = 24 - 8 * pass;
        uint which = pass & 1;
#ifndef QSA_SELECT_CHUNKS
        const uint b0 = 0;
#endif
        QSA_EACH_CHUNK {
            QSA_CHUNK_KEYS
            if (pass == 0) {
                // The top digit (sign and exponent) crowds a few bins: one
                // atomic per distinct bin per subgroup. The subgroup takes
                // its bins in ascending order, each lane counting all its
                // keys of the bin (Metal: the same count per key index of
                // the lanes, a loop over the bins for each).
                uint low = 0;
                while (true) {
                    uint next = 0xFFu;
                    QSA_EACH_KEY(QSA_TOP_NEXT)
                    uint leader = subgroupMin(next);
                    if (leader == 0xFFu) break;
                    uint count = 0;
                    QSA_EACH_KEY(QSA_TOP_COUNT)
                    count = subgroupAdd(count);
                    if (thread_index_in_simdgroup == 0) atomicAdd(hist[which][leader], count);
                    low = leader + 1;
                }
            } else {
                QSA_EACH_KEY(QSA_COUNT)
            }
        }
        barrier();
        // Thread t holds bin 255 - t: the scan gives the count above each
        // bin, and exactly one bin straddles the k_rem-th key.
        uint c = tid < 256 ? hist[which][255 - tid] : 0u;
        uint total;
        uint above = qsa_scan(c, total);
        if (tid < 256 && above < k_rem && above + c >= k_rem) {
            prefix_s = prefix | ((255 - tid) << shift);
            k_rem_s = k_rem - above;
        }
        if (tid < 256) hist[which][tid] = 0u;
        barrier();
        prefix = prefix_s;
        k_rem = k_rem_s;
        mask |= 0xFFu << shift;
    }
    uint threshold = prefix;
    // Elements above the threshold are all taken; exactly k_rem ties fill up.
    uint ties_to_take = k_rem;

    // Compaction in ascending block order, one scan per chunk: the low half
    // of the scanned value counts a thread's keys above the threshold, the
    // high half its ties (each at most CHUNK).
    uint sel_base = 0;
    uint tie_base = 0;
#ifndef QSA_SELECT_CHUNKS
    const uint b0 = 0;
#endif
    QSA_EACH_CHUNK {
        QSA_CHUNK_KEYS
        uint n_above = 0;
        uint n_tie = 0;
        QSA_EACH_KEY(QSA_TALLY)
        uint totals;
        uint ranks = qsa_scan((n_tie << 16) | n_above, totals);
        uint quota = ties_to_take > tie_base ? ties_to_take - tie_base : 0u;
        uint a = sel_base + (ranks & 0xFFFFu);
        uint t = ranks >> 16;
        QSA_EACH_KEY(QSA_PLACE)
        sel_base += (totals & 0xFFFFu) + min(totals >> 16, quota);
        tie_base += totals >> 16;
#ifdef QSA_SELECT_CHUNKS
        barrier();
#endif
    }
}
