// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The on-disk KV checkpoint format ("KVC"), byte-compatible with antirez's C ds4.
//!
//! We deliberately REUSE his format rather than invent one, because we already compile his
//! engine as `--engine reference` (build.rs pins antirez/ds4 glm52 @ bd89932). A shared format
//! means a checkpoint written by either engine can be read by the other, and one eviction /
//! budget policy governs one directory instead of two caches competing for the same disk.
//!
//! Layout, transcribed from `ds4_kvstore.c:fill_header` / `read_header` (rev bd89932). All
//! integers are LITTLE-ENDIAN. The fixed header is 48 bytes (`DS4_KVSTORE_FIXED_HEADER`),
//! followed by a 4-byte text length, then that many bytes of prompt text, then the payload.
//!
//! ```text
//!   0  'K'                        magic
//!   1  'V'
//!   2  'C'
//!   3  version            u8      KV_CACHE_VERSION = 1
//!   4  quant_bits         u8      2 or 4 — anything else is REJECTED on read
//!   5  reason             u8      see Reason; unknown values clamp to Unknown
//!   6  ext_flags          u8      bitset, see ExtFlags
//!   7  model_id           u8
//!   8  tokens             u32     0 is REJECTED on read
//!  12  hits               u32
//!  16  ctx_size           u32
//!  20  payload_abi        u8      KV_CACHE_PAYLOAD_ABI = 2 — mismatch REJECTED
//!  21..23                         padding (not written; zero)
//!  24  created_at         u64
//!  32  last_used          u64
//!  40  payload_bytes      u64
//!  48  text_bytes         u32     (immediately after the fixed header)
//!  52  text               [u8]
//!      payload            [u8]
//! ```
//!
//! Byte 20 is deliberately separate from the outer version: his comment notes the KVC envelope
//! can stay stable while the serialized session internals become unsafe to restore across
//! runtime changes. So a payload-ABI bump invalidates checkpoints without a format bump — and
//! since OUR payload is not his, we must not claim ABI 2 for a payload he could not restore.
//! See `PayloadOwner`.

pub const MAGIC: [u8; 3] = [b'K', b'V', b'C'];
pub const VERSION: u8 = 1;
pub const FIXED_HEADER: usize = 48;
/// Value antirez's engine writes. Ours must NOT reuse it for a different payload (see
/// `PayloadOwner::abi_byte`).
pub const PAYLOAD_ABI_ANTIREZ: u8 = 2;

/// Why a checkpoint was written. Values match the C enum exactly; unknown values clamp to
/// `Unknown` on read, as his reader does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    Unknown = 0,
    Cold = 1,
    Continued = 2,
    Evict = 3,
    Shutdown = 4,
    AgentSystem = 5,
    AgentSession = 6,
}

impl Reason {
    pub fn from_u8(v: u8) -> Reason {
        match v {
            1 => Reason::Cold,
            2 => Reason::Continued,
            3 => Reason::Evict,
            4 => Reason::Shutdown,
            5 => Reason::AgentSystem,
            6 => Reason::AgentSession,
            _ => Reason::Unknown,
        }
    }
}

/// `ds4_kvstore.h` ext-flag bitset.
pub mod ext_flags {
    pub const TOOL_MAP: u8 = 1 << 0;
    pub const RESPONSES_VISIBLE: u8 = 1 << 1;
    pub const THINKING_VISIBLE: u8 = 1 << 2;
    pub const SESSION_TITLE: u8 = 1 << 3;
}

/// Whose payload follows the header.
///
/// The envelope is shared; the PAYLOAD is not. His is a serialized `ds4_session` from his own
/// runtime — we cannot produce or consume it. Writing our payload under his ABI byte would
/// make his engine attempt a restore from bytes it cannot parse, which is worse than not
/// interoperating: the file passes every header check and fails at the point of use.
///
/// So our checkpoints carry a DIFFERENT payload ABI. His reader rejects them at `h[20]`
/// (`if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;`) — a clean, immediate refusal — while
/// ours reads both and only restores what it understands. Interop is therefore one-directional
/// and honest about it, rather than pretending at a compatibility we do not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadOwner {
    /// antirez's serialized session (payload ABI 2). We can read the header; not the payload.
    Antirez,
    /// Our Metal KV rings. ABI byte chosen so his reader refuses it up front.
    Ds4Rs,
}

impl PayloadOwner {
    pub fn abi_byte(self) -> u8 {
        match self {
            PayloadOwner::Antirez => PAYLOAD_ABI_ANTIREZ,
            // Deliberately not 2. High bit set so it can never collide with a future
            // upstream bump through simple increments.
            PayloadOwner::Ds4Rs => 0x80,
        }
    }
    pub fn from_abi(v: u8) -> Option<PayloadOwner> {
        match v {
            PAYLOAD_ABI_ANTIREZ => Some(PayloadOwner::Antirez),
            0x80 => Some(PayloadOwner::Ds4Rs),
            _ => None,
        }
    }
}

/// A parsed KVC header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub quant_bits: u8,
    pub reason: Reason,
    pub ext_flags: u8,
    pub model_id: u8,
    pub tokens: u32,
    pub hits: u32,
    pub ctx_size: u32,
    pub owner: PayloadOwner,
    pub created_at: u64,
    pub last_used: u64,
    pub payload_bytes: u64,
    pub text_bytes: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    TooShort,
    BadMagic,
    BadVersion(u8),
    /// Payload ABI we do not know — his reader refuses ours the same way.
    UnknownPayloadAbi(u8),
    /// His reader ends with `return e->tokens != 0 && (quant_bits == 2 || quant_bits == 4)`.
    ZeroTokens,
    BadQuantBits(u8),
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Parse the fixed header plus the 4-byte text length that follows it.
///
/// Applies exactly his acceptance rules, including the two that look like afterthoughts but
/// are load-bearing: a zero token count and a quant other than 2/4 are REJECTED, so a
/// truncated or foreign file cannot present as a valid empty checkpoint.
pub fn parse_header(buf: &[u8]) -> Result<Header, ParseError> {
    if buf.len() < FIXED_HEADER + 4 {
        return Err(ParseError::TooShort);
    }
    if buf[0..3] != MAGIC {
        return Err(ParseError::BadMagic);
    }
    if buf[3] != VERSION {
        return Err(ParseError::BadVersion(buf[3]));
    }
    let owner = PayloadOwner::from_abi(buf[20]).ok_or(ParseError::UnknownPayloadAbi(buf[20]))?;
    let tokens = le32(&buf[8..12]);
    if tokens == 0 {
        return Err(ParseError::ZeroTokens);
    }
    let quant_bits = buf[4];
    if quant_bits != 2 && quant_bits != 4 {
        return Err(ParseError::BadQuantBits(quant_bits));
    }
    Ok(Header {
        quant_bits,
        reason: Reason::from_u8(buf[5]),
        ext_flags: buf[6],
        model_id: buf[7],
        tokens,
        hits: le32(&buf[12..16]),
        ctx_size: le32(&buf[16..20]),
        owner,
        created_at: le64(&buf[24..32]),
        last_used: le64(&buf[32..40]),
        payload_bytes: le64(&buf[40..48]),
        text_bytes: le32(&buf[FIXED_HEADER..FIXED_HEADER + 4]),
    })
}

/// Serialize a header + text length, byte-identical to `ds4_kvstore_fill_header`.
pub fn write_header(h: &Header) -> Vec<u8> {
    let mut b = vec![0u8; FIXED_HEADER + 4];
    b[0..3].copy_from_slice(&MAGIC);
    b[3] = VERSION;
    b[4] = h.quant_bits;
    b[5] = h.reason as u8;
    b[6] = h.ext_flags;
    b[7] = h.model_id;
    b[8..12].copy_from_slice(&h.tokens.to_le_bytes());
    b[12..16].copy_from_slice(&h.hits.to_le_bytes());
    b[16..20].copy_from_slice(&h.ctx_size.to_le_bytes());
    b[20] = h.owner.abi_byte();
    // 21..24 padding stays zero — his fill_header never writes it.
    b[24..32].copy_from_slice(&h.created_at.to_le_bytes());
    b[32..40].copy_from_slice(&h.last_used.to_le_bytes());
    b[40..48].copy_from_slice(&h.payload_bytes.to_le_bytes());
    b[FIXED_HEADER..FIXED_HEADER + 4].copy_from_slice(&h.text_bytes.to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(owner: PayloadOwner) -> Header {
        Header {
            quant_bits: 4,
            reason: Reason::AgentSession,
            ext_flags: ext_flags::SESSION_TITLE | ext_flags::THINKING_VISIBLE,
            model_id: 7,
            tokens: 4096,
            hits: 3,
            ctx_size: 32768,
            owner,
            created_at: 1_784_000_000,
            last_used: 1_784_000_500,
            payload_bytes: 12_345_678,
            text_bytes: 91,
        }
    }

    /// Field OFFSETS are the contract with his C. A field written at the wrong offset still
    /// round-trips through our own code and is silently wrong to his reader, so pin the bytes.
    #[test]
    fn header_bytes_match_the_c_layout() {
        let b = write_header(&sample(PayloadOwner::Antirez));
        assert_eq!(b.len(), FIXED_HEADER + 4);
        assert_eq!(&b[0..3], b"KVC");
        assert_eq!(b[3], 1, "version");
        assert_eq!(b[4], 4, "quant_bits");
        assert_eq!(b[5], Reason::AgentSession as u8);
        assert_eq!(b[7], 7, "model_id");
        assert_eq!(le32(&b[8..12]), 4096, "tokens at offset 8");
        assert_eq!(le32(&b[12..16]), 3, "hits at offset 12");
        assert_eq!(le32(&b[16..20]), 32768, "ctx_size at offset 16");
        assert_eq!(b[20], PAYLOAD_ABI_ANTIREZ, "payload abi at offset 20");
        assert_eq!(&b[21..24], &[0, 0, 0], "padding must stay zero");
        assert_eq!(le64(&b[24..32]), 1_784_000_000, "created_at at 24");
        assert_eq!(le64(&b[32..40]), 1_784_000_500, "last_used at 32");
        assert_eq!(le64(&b[40..48]), 12_345_678, "payload_bytes at 40");
        assert_eq!(
            le32(&b[48..52]),
            91,
            "text_bytes immediately after the header"
        );
    }

    #[test]
    fn round_trips() {
        for owner in [PayloadOwner::Antirez, PayloadOwner::Ds4Rs] {
            let h = sample(owner);
            assert_eq!(parse_header(&write_header(&h)).unwrap(), h);
        }
    }

    /// OUR payload must not claim HIS ABI byte: his reader would pass every header check and
    /// then attempt to restore a serialized session from bytes that are not one. Refusing at
    /// the header is the honest failure.
    #[test]
    fn our_payload_never_claims_his_abi() {
        assert_ne!(PayloadOwner::Ds4Rs.abi_byte(), PAYLOAD_ABI_ANTIREZ);
        let ours = write_header(&sample(PayloadOwner::Ds4Rs));
        // Emulate HIS check: `if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;`
        assert_ne!(
            ours[20], PAYLOAD_ABI_ANTIREZ,
            "his reader must refuse our file up front"
        );
    }

    /// His two trailing acceptance rules are load-bearing: without them a truncated or
    /// foreign file can present as a valid empty checkpoint.
    #[test]
    fn rejects_what_the_c_rejects() {
        let good = write_header(&sample(PayloadOwner::Antirez));

        let mut zero_tokens = good.clone();
        zero_tokens[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(parse_header(&zero_tokens), Err(ParseError::ZeroTokens));

        let mut bad_quant = good.clone();
        bad_quant[4] = 3;
        assert_eq!(parse_header(&bad_quant), Err(ParseError::BadQuantBits(3)));

        let mut bad_magic = good.clone();
        bad_magic[1] = b'X';
        assert_eq!(parse_header(&bad_magic), Err(ParseError::BadMagic));

        let mut bad_ver = good.clone();
        bad_ver[3] = 2;
        assert_eq!(parse_header(&bad_ver), Err(ParseError::BadVersion(2)));

        let mut bad_abi = good.clone();
        bad_abi[20] = 9;
        assert_eq!(
            parse_header(&bad_abi),
            Err(ParseError::UnknownPayloadAbi(9))
        );

        assert_eq!(
            parse_header(&good[..FIXED_HEADER]),
            Err(ParseError::TooShort)
        );
    }

    /// Unknown reason codes clamp rather than fail — his reader does the same, so a file from
    /// a newer upstream stays readable instead of being discarded.
    #[test]
    fn unknown_reason_clamps_like_the_c() {
        assert_eq!(Reason::from_u8(200), Reason::Unknown);
        assert_eq!(Reason::from_u8(6), Reason::AgentSession);
    }
}

// ---------------------------------------------------------------------------
// Payload descriptor (ours — payload ABI 0x80)
//
// The envelope says a checkpoint is well-formed. It does NOT say the bytes are safe to load
// into THIS runtime. A KV ring is meaningless outside the exact geometry that produced it:
// restore rings from a different model, quant, head count or head dim and nothing errors --
// attention simply reads plausible numbers and generates subtly wrong text. That failure has
// no loud symptom, so every fact needed to rule it out is recorded here and checked before a
// single byte of ring data is trusted.

/// Descriptor prefixing our payload. Fixed size, little-endian, same conventions as the
/// envelope so one hexdump reads end to end.
pub const DESCRIPTOR_BYTES: usize = 48;
/// Bumped whenever the RING ENCODING changes in a way older files cannot satisfy. Distinct
/// from the envelope version and from the payload-ABI byte: those say "whose payload and what
/// container", this says "which encoding of ours".
pub const DESCRIPTOR_VERSION: u16 = 1;

/// How KV entries are stored in the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KvDtype {
    F16 = 1,
    Q8 = 2,
}

impl KvDtype {
    pub fn from_u8(v: u8) -> Option<KvDtype> {
        match v {
            1 => Some(KvDtype::F16),
            2 => Some(KvDtype::Q8),
            _ => None,
        }
    }
    /// Bytes per stored element.
    pub fn elem_bytes(self) -> u64 {
        match self {
            KvDtype::F16 => 2,
            KvDtype::Q8 => 1,
        }
    }
}

/// Everything that must match before a restore is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Descriptor {
    pub version: u16,
    /// Identity of the weights the rings were built from. A restore across models is the
    /// most dangerous mismatch and the least visible, so this is checked first.
    pub model_fingerprint: u64,
    pub quant_bits: u8,
    pub kv_dtype: KvDtype,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Ring capacity in tokens.
    pub ctx_size: u32,
    /// Valid tokens currently in the ring. Never greater than `ctx_size`.
    pub n_pos: u32,
}

impl Descriptor {
    /// `None` if the geometry's implied size does not fit in a `u64`.
    ///
    /// The four dimensions are `u32` read straight out of an untrusted file, so their product
    /// can reach ~2^130 — far past `u64`. Unchecked, that wraps: `2 * 2^16 * 2^16 * 2^16 *
    /// 2^16 * 1 == 2^65 ≡ 0 (mod 2^64)`, i.e. four *nonzero* dimensions producing a 0-byte
    /// expected size and defeating the [`RestoreError::ZeroGeometry`] guard below (in a debug
    /// build it traps instead, turning corrupt input into a panic). Checked throughout.
    pub fn checked_payload_bytes(&self) -> Option<u64> {
        2u64 // K and V
            .checked_mul(self.n_layers as u64)?
            .checked_mul(self.ctx_size as u64)?
            .checked_mul(self.n_kv_heads as u64)?
            .checked_mul(self.head_dim as u64)?
            .checked_mul(self.kv_dtype.elem_bytes())
    }

    /// Exact size of the K+V ring data this descriptor implies.
    ///
    /// Checked against the actual payload length on restore: a descriptor that agrees with the
    /// runtime but a payload that is short means truncation (a partial write, a full disk),
    /// which would otherwise be read as valid rings followed by garbage.
    ///
    /// `parse` rejects any geometry that cannot be represented, so for a parsed descriptor this
    /// is exact. The fields are `pub`, though, so a hand-built descriptor can still overflow —
    /// that case saturates to `u64::MAX`, which fails CLOSED (no real payload is that long, so
    /// the size check rejects) rather than wrapping to a small number that might match.
    pub fn expected_payload_bytes(&self) -> u64 {
        self.checked_payload_bytes().unwrap_or(u64::MAX)
    }

    pub fn write(&self) -> Vec<u8> {
        let mut b = vec![0u8; DESCRIPTOR_BYTES];
        b[0..2].copy_from_slice(&self.version.to_le_bytes());
        b[2] = self.quant_bits;
        b[3] = self.kv_dtype as u8;
        b[4..8].copy_from_slice(&self.n_layers.to_le_bytes());
        b[8..12].copy_from_slice(&self.n_kv_heads.to_le_bytes());
        b[12..16].copy_from_slice(&self.head_dim.to_le_bytes());
        b[16..20].copy_from_slice(&self.ctx_size.to_le_bytes());
        b[20..24].copy_from_slice(&self.n_pos.to_le_bytes());
        b[24..32].copy_from_slice(&self.model_fingerprint.to_le_bytes());
        // 32..48 reserved, zero — room to add a field without a version bump.
        b
    }

    pub fn parse(buf: &[u8]) -> Result<Descriptor, RestoreError> {
        if buf.len() < DESCRIPTOR_BYTES {
            return Err(RestoreError::DescriptorTooShort);
        }
        let version = u16::from_le_bytes([buf[0], buf[1]]);
        if version != DESCRIPTOR_VERSION {
            return Err(RestoreError::DescriptorVersion(version));
        }
        let kv_dtype = KvDtype::from_u8(buf[3]).ok_or(RestoreError::UnknownKvDtype(buf[3]))?;
        let d = Descriptor {
            version,
            quant_bits: buf[2],
            kv_dtype,
            n_layers: le32(&buf[4..8]),
            n_kv_heads: le32(&buf[8..12]),
            head_dim: le32(&buf[12..16]),
            ctx_size: le32(&buf[16..20]),
            n_pos: le32(&buf[20..24]),
            model_fingerprint: le64(&buf[24..32]),
        };
        // A zero dimension makes expected_payload_bytes() collapse to 0, which would then
        // "match" an empty payload and restore an empty ring as though it were valid.
        if d.n_layers == 0 || d.n_kv_heads == 0 || d.head_dim == 0 || d.ctx_size == 0 {
            return Err(RestoreError::ZeroGeometry);
        }
        // ...and the same invariant against the PRODUCT, not just each factor: a size that
        // does not fit in u64 would wrap (potentially to 0) and masquerade as a short ring.
        if d.checked_payload_bytes().is_none() {
            return Err(RestoreError::GeometryOverflow);
        }
        if d.n_pos > d.ctx_size {
            return Err(RestoreError::PosBeyondCtx {
                n_pos: d.n_pos,
                ctx: d.ctx_size,
            });
        }
        Ok(d)
    }

    /// May a checkpoint described by `self` be restored into the runtime described by `live`?
    ///
    /// `n_pos` is deliberately NOT compared: it is the checkpoint's contents, not its shape.
    /// Everything else must be identical -- there is no "close enough" geometry, and no
    /// mismatch here is recoverable by loading anyway.
    pub fn check_compatible(&self, live: &Descriptor) -> Result<(), RestoreError> {
        if self.model_fingerprint != live.model_fingerprint {
            return Err(RestoreError::ModelMismatch {
                file: self.model_fingerprint,
                live: live.model_fingerprint,
            });
        }
        if self.quant_bits != live.quant_bits {
            return Err(RestoreError::QuantMismatch {
                file: self.quant_bits,
                live: live.quant_bits,
            });
        }
        if self.kv_dtype != live.kv_dtype {
            return Err(RestoreError::DtypeMismatch);
        }
        for (what, a, b) in [
            ("n_layers", self.n_layers, live.n_layers),
            ("n_kv_heads", self.n_kv_heads, live.n_kv_heads),
            ("head_dim", self.head_dim, live.head_dim),
            ("ctx_size", self.ctx_size, live.ctx_size),
        ] {
            if a != b {
                return Err(RestoreError::GeometryMismatch {
                    what,
                    file: a,
                    live: b,
                });
            }
        }
        Ok(())
    }

    /// Full gate: parse, confirm it fits this runtime, and confirm the payload is all there.
    /// Nothing downstream should touch ring bytes without this returning Ok.
    pub fn validate_restore(payload: &[u8], live: &Descriptor) -> Result<Descriptor, RestoreError> {
        let d = Descriptor::parse(payload)?;
        d.check_compatible(live)?;
        let want = d.expected_payload_bytes();
        let have = payload.len() as u64 - DESCRIPTOR_BYTES as u64;
        if have != want {
            return Err(RestoreError::PayloadSize { want, have });
        }
        Ok(d)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RestoreError {
    DescriptorTooShort,
    DescriptorVersion(u16),
    UnknownKvDtype(u8),
    ZeroGeometry,
    /// Geometry whose implied payload size does not fit in a u64 (would wrap).
    GeometryOverflow,
    PosBeyondCtx {
        n_pos: u32,
        ctx: u32,
    },
    ModelMismatch {
        file: u64,
        live: u64,
    },
    QuantMismatch {
        file: u8,
        live: u8,
    },
    DtypeMismatch,
    GeometryMismatch {
        what: &'static str,
        file: u32,
        live: u32,
    },
    PayloadSize {
        want: u64,
        have: u64,
    },
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestoreError::DescriptorTooShort => write!(f, "checkpoint payload is truncated"),
            RestoreError::DescriptorVersion(v) => {
                write!(
                    f,
                    "checkpoint uses ring encoding v{v}, this build writes v{DESCRIPTOR_VERSION}"
                )
            }
            RestoreError::UnknownKvDtype(v) => write!(f, "unknown KV dtype {v}"),
            RestoreError::ZeroGeometry => write!(f, "checkpoint declares a zero dimension"),
            RestoreError::GeometryOverflow => {
                write!(
                    f,
                    "checkpoint declares a geometry too large to be a real ring"
                )
            }
            RestoreError::PosBeyondCtx { n_pos, ctx } => {
                write!(f, "checkpoint holds {n_pos} tokens in a {ctx}-token ring")
            }
            RestoreError::ModelMismatch { .. } => {
                write!(f, "checkpoint belongs to a different model")
            }
            RestoreError::QuantMismatch { file, live } => {
                write!(f, "checkpoint is q{file}, this model is q{live}")
            }
            RestoreError::DtypeMismatch => write!(f, "checkpoint KV dtype differs"),
            RestoreError::GeometryMismatch { what, file, live } => {
                write!(f, "{what} differs: checkpoint {file}, this model {live}")
            }
            RestoreError::PayloadSize { want, have } => {
                write!(
                    f,
                    "checkpoint is {have} bytes, expected {want} — truncated or corrupt"
                )
            }
        }
    }
}

#[cfg(test)]
mod descriptor_tests {
    use super::*;

    fn live() -> Descriptor {
        Descriptor {
            version: DESCRIPTOR_VERSION,
            model_fingerprint: 0xDEAD_BEEF_CAFE_0001,
            quant_bits: 4,
            kv_dtype: KvDtype::F16,
            n_layers: 28,
            n_kv_heads: 4,
            head_dim: 128,
            ctx_size: 4096,
            n_pos: 1024,
        }
    }

    fn payload_for(d: &Descriptor) -> Vec<u8> {
        let mut v = d.write();
        v.resize(DESCRIPTOR_BYTES + d.expected_payload_bytes() as usize, 0);
        v
    }

    #[test]
    fn round_trips_and_pins_offsets() {
        let d = live();
        let b = d.write();
        assert_eq!(b.len(), DESCRIPTOR_BYTES);
        assert_eq!(u16::from_le_bytes([b[0], b[1]]), DESCRIPTOR_VERSION);
        assert_eq!(b[2], 4, "quant_bits");
        assert_eq!(b[3], KvDtype::F16 as u8);
        assert_eq!(le32(&b[16..20]), 4096, "ctx_size at 16");
        assert_eq!(le64(&b[24..32]), 0xDEAD_BEEF_CAFE_0001, "fingerprint at 24");
        assert_eq!(&b[32..48], &[0u8; 16], "reserved stays zero");
        assert_eq!(Descriptor::parse(&b).unwrap(), d);
    }

    /// THE reason this type exists: a KV ring from another model loads without error and
    /// produces subtly wrong text. It must be refused, not tolerated.
    #[test]
    fn a_checkpoint_from_another_model_is_refused() {
        let mut file = live();
        file.model_fingerprint = 0x1111_2222_3333_4444;
        let err = Descriptor::validate_restore(&payload_for(&file), &live()).unwrap_err();
        assert!(matches!(err, RestoreError::ModelMismatch { .. }), "{err:?}");
        assert!(err.to_string().contains("different model"));
    }

    /// Every geometry field is load-bearing; none may be "close enough".
    #[test]
    fn every_geometry_difference_is_refused() {
        for (name, mutate) in [
            (
                "n_layers",
                (|d: &mut Descriptor| d.n_layers += 1) as fn(&mut Descriptor),
            ),
            ("n_kv_heads", |d: &mut Descriptor| d.n_kv_heads += 1),
            ("head_dim", |d: &mut Descriptor| d.head_dim += 1),
            ("ctx_size", |d: &mut Descriptor| d.ctx_size *= 2),
        ] {
            let mut file = live();
            mutate(&mut file);
            let err = Descriptor::validate_restore(&payload_for(&file), &live()).unwrap_err();
            assert!(
                matches!(err, RestoreError::GeometryMismatch { .. }),
                "{name} mismatch was not refused: {err:?}"
            );
        }
        // quant and dtype too
        let mut q = live();
        q.quant_bits = 2;
        assert!(matches!(
            Descriptor::validate_restore(&payload_for(&q), &live()).unwrap_err(),
            RestoreError::QuantMismatch { .. }
        ));
        let mut t = live();
        t.kv_dtype = KvDtype::Q8;
        assert!(matches!(
            Descriptor::validate_restore(&payload_for(&t), &live()).unwrap_err(),
            RestoreError::DtypeMismatch
        ));
    }

    /// A short payload is truncation (partial write, full disk). Without this it reads as
    /// valid rings followed by garbage.
    #[test]
    fn a_truncated_payload_is_refused() {
        let d = live();
        let mut p = payload_for(&d);
        p.truncate(p.len() - 1);
        let err = Descriptor::validate_restore(&p, &d).unwrap_err();
        assert!(matches!(err, RestoreError::PayloadSize { .. }), "{err:?}");
        assert!(err.to_string().contains("truncated or corrupt"));
    }

    /// A zero dimension would make the expected size collapse to 0 and "match" an empty
    /// payload — an empty ring restored as though valid.
    #[test]
    fn zero_geometry_cannot_masquerade_as_an_empty_ring() {
        let mut d = live();
        d.n_kv_heads = 0;
        assert_eq!(
            Descriptor::parse(&d.write()),
            Err(RestoreError::ZeroGeometry)
        );
    }

    /// n_pos is CONTENT, not shape: an identical-geometry checkpoint holding a different
    /// number of tokens is exactly what resuming a session looks like.
    #[test]
    fn a_different_token_count_is_allowed() {
        let mut file = live();
        file.n_pos = 77;
        assert!(Descriptor::validate_restore(&payload_for(&file), &live()).is_ok());
        // ...but more tokens than the ring can hold is corrupt.
        let mut bad = live();
        bad.n_pos = bad.ctx_size + 1;
        assert!(matches!(
            Descriptor::parse(&bad.write()),
            Err(RestoreError::PosBeyondCtx { .. })
        ));
    }

    #[test]
    fn expected_size_matches_the_ring_arithmetic() {
        let d = live();
        // 2 (K+V) * 28 layers * 4096 ctx * 4 heads * 128 dim * 2 bytes
        assert_eq!(d.expected_payload_bytes(), 2 * 28 * 4096 * 4 * 128 * 2);
    }

    /// The `ZeroGeometry` guard states an invariant — "no accepted descriptor yields an
    /// expected size of 0" — but enforced it only on each dimension INDIVIDUALLY. The size is
    /// a product of four u32s in u64 space, and four large-but-nonzero dimensions overflow it:
    /// 2 * 2^16 * 2^16 * 2^16 * 2^16 * 1 == 2^65 ≡ 0 (mod 2^64).
    ///
    /// So a descriptor with every dimension nonzero parsed as valid and then reported an
    /// expected payload of 0 bytes — re-opening precisely the hole `ZeroGeometry` exists to
    /// close, with an empty payload "matching" and an empty ring restored as though valid.
    /// (In a debug build the wrap traps instead: `attempt to multiply with overflow`. Either
    /// way, untrusted bytes drive it.)
    #[test]
    fn overflowing_geometry_cannot_masquerade_as_an_empty_ring() {
        let mut d = live();
        d.kv_dtype = KvDtype::Q8;
        d.n_layers = 1 << 16;
        d.n_kv_heads = 1 << 16;
        d.head_dim = 1 << 16;
        d.ctx_size = 1 << 16;
        d.n_pos = 0;

        // Every dimension is nonzero, so the per-dimension guard has nothing to say.
        assert_eq!(
            Descriptor::parse(&d.write()),
            Err(RestoreError::GeometryOverflow)
        );

        // And the size accessor must never wrap to a small number: it fails CLOSED, so a
        // caller that skips parse() still cannot be tricked into matching a short payload.
        assert_eq!(d.checked_payload_bytes(), None);
        assert_eq!(d.expected_payload_bytes(), u64::MAX);
    }

    /// The invariant, stated directly: no descriptor that `parse` ACCEPTS may report an
    /// expected payload of zero. Swept over the full u32 exponent range rather than the one
    /// hand-picked value above, so a future change to the bound cannot quietly re-admit a
    /// wrapping geometry.
    #[test]
    fn no_accepted_descriptor_ever_reports_zero_expected_bytes() {
        let mut accepted = 0usize;
        for dtype in [KvDtype::Q8, KvDtype::F16] {
            for a in 0..32u32 {
                for b in 0..32u32 {
                    let mut d = live();
                    d.kv_dtype = dtype;
                    d.n_layers = 1u32 << a;
                    d.n_kv_heads = 1u32 << b;
                    d.head_dim = 1u32 << a;
                    d.ctx_size = 1u32 << b;
                    d.n_pos = 0;
                    if let Ok(parsed) = Descriptor::parse(&d.write()) {
                        accepted += 1;
                        let n = parsed.expected_payload_bytes();
                        assert_ne!(
                            n, 0,
                            "parse accepted layers=2^{a} heads=2^{b} dim=2^{a} ctx=2^{b} \
                             dtype={dtype:?} yet it claims a 0-byte payload"
                        );
                        assert_eq!(parsed.checked_payload_bytes(), Some(n));
                    }
                }
            }
        }
        // Discrimination guard: a sweep that accepted nothing would pass vacuously.
        assert!(accepted > 100, "sweep accepted only {accepted} descriptors");
    }
}
