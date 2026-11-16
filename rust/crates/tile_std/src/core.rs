// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// FIXME: In order to test codegen for kernels written in Rust we only support a small subset
// of the `core` library. This file have to be removed once backend is advanced enough to
// build the whole `core`.

// The crate root (lib.rs) already carries `#![rustc_coherence_is_core]`. Older
// nightlies also accepted it on this module; the 1.99 cycle restricts it to the
// crate root (build.rs `rustc_coherence_root_only`), so omit the redundant
// module-level copy there.
#![cfg_attr(not(rustc_coherence_root_only), rustc_coherence_is_core)]

pub mod marker {
    // region:sized
    #[lang = "pointee_sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait PointeeSized {}

    #[lang = "meta_sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait MetaSized: PointeeSized {}

    #[lang = "sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait Sized: MetaSized {}
    // endregion:sized

    // region:send
    pub unsafe auto trait Send {}

    impl<T: PointeeSized> !Send for *const T {}
    impl<T: PointeeSized> !Send for *mut T {}
    // region:sync
    unsafe impl<T: Sync + PointeeSized> Send for &T {}
    unsafe impl<T: Send + PointeeSized> Send for &mut T {}
    // endregion:sync
    // endregion:send

    // region:sync
    #[lang = "sync"]
    pub unsafe auto trait Sync {}

    impl<T: PointeeSized> !Sync for *const T {}
    impl<T: PointeeSized> !Sync for *mut T {}
    // endregion:sync

    // region:unsize
    #[lang = "unsize"]
    pub trait Unsize<T: PointeeSized>: PointeeSized {}
    // endregion:unsize

    // region:unpin
    #[lang = "unpin"]
    pub auto trait Unpin {}
    // endregion:unpin

    // region:copy
    #[lang = "copy"]
    pub trait Copy: Clone {}
    // region:derive
    #[rustc_builtin_macro]
    pub macro Copy($item:item) {}
    // endregion:derive

    pub mod copy_impls {
        use super::{Copy, PointeeSized};

        macro_rules! impl_copy {
            ($($t:ty)*) => {
                $(
                    impl Copy for $t {}
                )*
            }
        }

        impl_copy! {
            usize u8 u16 u32 u64
            isize i8 i16 i32 i64
            f16 f32 f128
            bool char
        }

        impl<T: PointeeSized> Copy for *const T {}
        impl<T: PointeeSized> Copy for *mut T {}
        impl<T: PointeeSized> Copy for &T {}
        impl Copy for ! {}
    }
    // endregion:copy

    // region:tuple
    #[lang = "tuple_trait"]
    pub trait Tuple {}
    // endregion:tuple

    // region:phantom_data
    #[lang = "phantom_data"]
    pub struct PhantomData<T: PointeeSized>;

    // region:clone
    impl<T: PointeeSized> Clone for PhantomData<T> {
        fn clone(&self) -> Self {
            Self
        }
    }
    // endregion:clone

    // region:copy
    impl<T: PointeeSized> Copy for PhantomData<T> {}
    // endregion:copy

    // endregion:phantom_data

    // region:discriminant
    #[lang = "discriminant_kind"]
    pub trait DiscriminantKind {
        #[lang = "discriminant_type"]
        type Discriminant;
    }
    // endregion:discriminant

    // region:coerce_pointee
    #[rustc_builtin_macro(CoercePointee, attributes(pointee))]
    #[allow_internal_unstable(dispatch_from_dyn, coerce_unsized, unsize)]
    pub macro CoercePointee($item:item) {
        /* compiler built-in */
    }
    // endregion:coerce_pointee

    #[lang = "destruct"]
    #[rustc_deny_explicit_impl]
    // This marker attribute was renamed `rustc_do_not_implement_via_object` ->
    // `rustc_dyn_incompatible_trait` within the 2026-02 nightly cycle. Gate on the
    // `build.rs`-detected commit date so tile_std builds on both the pinned
    // 2025-08-04 toolchain and current nightlies.
    #[cfg_attr(rustc_dyn_incompatible_trait_attr, rustc_dyn_incompatible_trait)]
    #[cfg_attr(not(rustc_dyn_incompatible_trait_attr), rustc_do_not_implement_via_object)]
    pub trait Destruct: PointeeSized {}

    #[lang = "freeze"]
    pub unsafe auto trait Freeze {}

    #[lang = "structural_peq"]
    pub trait StructuralPartialEq {}
}

// region:default
pub mod default {
    pub trait Default: Sized {
        fn default() -> Self;
    }
    // region:derive
    #[rustc_builtin_macro(Default, attributes(default))]
    pub macro Default($item:item) {}
    // endregion:derive

    // region:builtin_impls
    macro_rules! impl_default {
        ($v:literal; $($t:ty)*) => {
            $(
                impl Default for $t {
                    fn default() -> Self {
                        $v
                    }
                }
            )*
        }
    }

    impl_default! {
        0; usize u8 u16 u32 u64 isize i8 i16 i32 i64
    }
    impl_default! {
        0.0; f16 f32 f128
    }
    // endregion:builtin_impls
}
// endregion:default

// region:hash
pub mod hash {
    use crate::core::marker::PointeeSized;

    pub trait Hasher {}

    pub trait Hash: PointeeSized {
        fn hash<H: Hasher>(&self, state: &mut H);
    }

    // region:derive
    pub mod derive {
        #[rustc_builtin_macro]
        pub macro Hash($item:item) {}
    }
    pub use derive::Hash;
    // endregion:derive
}
// endregion:hash

// region:cell
pub mod cell {
    use crate::core::marker::PointeeSized;
    use crate::core::mem;

    #[lang = "unsafe_cell"]
    pub struct UnsafeCell<T: PointeeSized> {
        value: T,
    }

    impl<T> UnsafeCell<T> {
        pub const fn new(value: T) -> UnsafeCell<T> {
            UnsafeCell { value }
        }

        pub const fn get(&self) -> *mut T {
            self as *const UnsafeCell<T> as *const T as *mut T
        }
    }

    pub struct Cell<T: PointeeSized> {
        value: UnsafeCell<T>,
    }

    impl<T> Cell<T> {
        pub const fn new(value: T) -> Cell<T> {
            Cell {
                value: UnsafeCell::new(value),
            }
        }

        pub fn set(&self, val: T) {
            let old = self.replace(val);
            mem::drop(old);
        }

        pub fn replace(&self, val: T) -> T {
            mem::replace(unsafe { &mut *self.value.get() }, val)
        }
    }

    impl<T: Copy> Cell<T> {
        pub fn get(&self) -> T {
            unsafe { *self.value.get() }
        }
    }
}
// endregion:cell

// region:clone
pub mod clone {
    #[lang = "clone"]
    pub trait Clone: Sized {
        fn clone(&self) -> Self;
    }

    // The built-in `#[derive(Clone)]` expansion references
    // `core::clone::TrivialClone` on nightlies after 2025-08-04 to fast-path
    // field-trivial clones. This `no_core` crate must supply the marker itself.
    pub unsafe trait TrivialClone: Clone {}

    // region:builtin_impls
    macro_rules! impl_clone {
        ($($t:ty)*) => {
            $(
                impl Clone for $t {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
            )*
        }
    }

    impl_clone! {
        usize u8 u16 u32 u64
        isize i8 i16 i32 i64
        f16 f32 f128
        bool char
    }

    impl Clone for ! {
        fn clone(&self) -> ! {
            loop {}
        }
    }

    impl<T: Clone> Clone for [T; 0] {
        fn clone(&self) -> Self {
            []
        }
    }

    impl<T: Clone> Clone for [T; 1] {
        fn clone(&self) -> Self {
            [self[0].clone()]
        }
    }
    // endregion:builtin_impls

    // region:derive
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}
    // endregion:derive

    use crate::marker::PointeeSized;
    pub struct AssertParamIsClone<T: Clone + PointeeSized> {
        _field: crate::core::marker::PhantomData<T>,
    }

    pub struct AssertParamIsCopy<T: Copy + PointeeSized> {
        _field: crate::marker::PhantomData<T>,
    }

    impl<T: PointeeSized> Clone for *const T {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<T: PointeeSized> Clone for *mut T {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    /// Shared references can be cloned, but mutable references *cannot*!
    impl<T: PointeeSized> Clone for &T {
        #[inline(always)]
        #[rustc_diagnostic_item = "noop_method_clone"]
        fn clone(&self) -> Self {
            self
        }
    }

    /// Shared references can be cloned, but mutable references *cannot*!
    impl<T: PointeeSized> !Clone for &mut T {}
}
// endregion:clone

pub mod convert {
    // region:from
    pub trait From<T>: Sized {
        fn from(_: T) -> Self;
    }
    pub trait Into<T>: Sized {
        fn into(self) -> T;
    }

    impl<T, U> Into<U> for T
    where
        U: From<T>,
    {
        fn into(self) -> U {
            U::from(self)
        }
    }

    impl<T> From<T> for T {
        fn from(t: T) -> T {
            t
        }
    }

    pub trait TryFrom<T>: Sized {
        type Error;
        fn try_from(value: T) -> Result<Self, Self::Error>;
    }
    pub trait TryInto<T>: Sized {
        type Error;
        fn try_into(self) -> Result<T, Self::Error>;
    }

    impl<T, U> TryInto<U> for T
    where
        U: TryFrom<T>,
    {
        type Error = U::Error;
        fn try_into(self) -> Result<U, U::Error> {
            U::try_from(self)
        }
    }
    // endregion:from

    // region:as_ref
    pub trait AsRef<T: crate::core::marker::PointeeSized>:
        crate::core::marker::PointeeSized
    {
        fn as_ref(&self) -> &T;
    }
    // endregion:as_ref
    // region:as_mut
    pub trait AsMut<T: crate::core::marker::PointeeSized>:
        crate::core::marker::PointeeSized
    {
        fn as_mut(&mut self) -> &mut T;
    }
    // endregion:as_mut
    // region:infallible
    pub enum Infallible {}
    // endregion:infallible
}

pub mod borrow {
    // region:borrow
    pub trait Borrow<Borrowed: ?Sized> {
        fn borrow(&self) -> &Borrowed;
    }
    // endregion:borrow

    // region:borrow_mut
    pub trait BorrowMut<Borrowed: ?Sized>: Borrow<Borrowed> {
        fn borrow_mut(&mut self) -> &mut Borrowed;
    }
    // endregion:borrow_mut
}

pub mod mem {
    // region:manually_drop
    use crate::core::marker::PointeeSized;

    #[lang = "manually_drop"]
    #[repr(transparent)]
    pub struct ManuallyDrop<T: PointeeSized> {
        value: T,
    }

    impl<T> ManuallyDrop<T> {
        pub const fn new(value: T) -> ManuallyDrop<T> {
            ManuallyDrop { value }
        }

        pub const fn into_inner(slot: ManuallyDrop<T>) -> T {
            slot.value
        }
    }

    impl<T: Copy> Copy for ManuallyDrop<T> {}
    impl<T: Clone> Clone for ManuallyDrop<T> {
        fn clone(&self) -> Self {
            ManuallyDrop::new(self.value.clone())
        }
    }

    // region:deref
    // Use `?Sized` (relaxes to PointeeSized) to match the Deref::Target bound;
    // an explicit `PointeeSized` bound instead over-constrains to MetaSized here.
    impl<T: ?Sized> crate::core::ops::Deref for ManuallyDrop<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.value
        }
    }
    // endregion:deref

    // endregion:manually_drop

    // region:drop
    pub fn drop<T>(_x: T) {}
    pub fn swap<T>(x: &mut T, y: &mut T) {
        // Byte-swap x and y without requiring T: Copy
        unsafe {
            let size = size_of::<T>();
            let xp = x as *mut T as *mut u8;
            let yp = y as *mut T as *mut u8;
            let mut i = 0usize;
            while i < size {
                let tmp = *xp.wrapping_add(i);
                *xp.wrapping_add(i) = *yp.wrapping_add(i);
                *yp.wrapping_add(i) = tmp;
                i = i + 1;
            }
        }
    }
    pub fn replace<T>(dest: &mut T, mut src: T) -> T {
        // Byte-swap dest and src without requiring T: Copy
        unsafe {
            let size = size_of::<T>();
            let dp = dest as *mut T as *mut u8;
            let sp = &mut src as *mut T as *mut u8;
            let mut i = 0usize;
            while i < size {
                let tmp = *dp.wrapping_add(i);
                *dp.wrapping_add(i) = *sp.wrapping_add(i);
                *sp.wrapping_add(i) = tmp;
                i = i + 1;
            }
        }
        src // now contains the old value from dest
    }
    // endregion:drop

    // region:transmute
    #[rustc_intrinsic]
    pub const unsafe fn transmute<Src, Dst>(src: Src) -> Dst;
    // endregion:transmute

    // region:size_of
    #[rustc_intrinsic]
    pub const fn size_of<T>() -> usize;

    #[rustc_intrinsic]
    pub const fn align_of<T>() -> usize;

    pub const fn size_of_val<T>(val: &T) -> usize {
        size_of::<T>()
    }

    pub const fn align_of_val<T>(val: &T) -> usize {
        align_of::<T>()
    }

    pub fn forget<T>(t: T) {
        let _ = ManuallyDrop::new(t);
    }
    // endregion:size_of

    // region:maybe_uninit
    // NOTE: Implemented as a struct wrapping ManuallyDrop instead of a union
    // because on nightly-2025-08-04 the compiler does not recognize our
    // #[lang = "manually_drop"] for the union field Copy exemption.
    #[repr(transparent)]
    pub struct MaybeUninit<T> {
        value: ManuallyDrop<T>,
    }

    impl<T> MaybeUninit<T> {
        pub fn uninit() -> MaybeUninit<T> {
            // In NPU kernel context, uninit values are allocated by the runtime.
            // This stub exists for API compatibility.
            loop {}
        }

        pub const fn new(val: T) -> MaybeUninit<T> {
            MaybeUninit {
                value: ManuallyDrop::new(val),
            }
        }

        pub unsafe fn assume_init(self) -> T {
            unsafe {
                crate::core::intrinsics::assert_inhabited::<T>();
                ManuallyDrop::into_inner(self.value)
            }
        }

        pub unsafe fn assume_init_ref(&self) -> &T {
            unsafe { &*self.as_ptr() }
        }

        pub const fn as_ptr(&self) -> *const T {
            self as *const _ as *const T
        }

        pub fn as_mut_ptr(&mut self) -> *mut T {
            self as *mut _ as *mut T
        }

        pub fn write(&mut self, val: T) -> &mut T {
            *self = MaybeUninit::new(val);
            unsafe { self.assume_init_mut() }
        }

        pub unsafe fn assume_init_mut(&mut self) -> &mut T {
            unsafe { &mut *self.as_mut_ptr() }
        }
    }

    impl<T: Copy> Clone for MaybeUninit<T> {
        fn clone(&self) -> Self {
            MaybeUninit {
                value: self.value.clone(),
            }
        }
    }

    impl<T: Copy> Copy for MaybeUninit<T> {}
    // endregion:maybe_uninit

    // region:discriminant
    use crate::core::marker::DiscriminantKind;
    pub struct Discriminant<T>(<T as DiscriminantKind>::Discriminant);
    // endregion:discriminant

    // region:offset_of
    pub macro offset_of($Container:ty, $($fields:expr)+ $(,)?) {
        // The `{}` is for better error messages
        {builtin # offset_of($Container, $($fields)+)}
    }
    // endregion:offset_of
}

pub mod ptr {
    use crate::core::hash::Hash;
    use crate::core::marker::PointeeSized;
    use crate::marker::Unpin;

    // region:drop
    // The `drop_in_place` lang item was renamed to `drop_glue` (build.rs `rustc_drop_glue_lang`).
    #[cfg_attr(rustc_drop_glue_lang, lang = "drop_glue")]
    #[cfg_attr(not(rustc_drop_glue_lang), lang = "drop_in_place")]
    pub unsafe fn drop_in_place<T: PointeeSized>(_to_drop: *mut T) {
        // unsafe { drop_in_place(to_drop) }
    }
    pub unsafe fn read<T: Copy>(src: *const T) -> T {
        unsafe { *src }
    }
    pub unsafe fn write<T>(dst: *mut T, src: T) {
        unsafe {
            *dst = src;
        }
    }
    pub unsafe fn read_volatile<T: Copy>(src: *const T) -> T {
        unsafe { *src }
    }
    pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
        unsafe {
            *dst = src;
        }
    }
    // endregion:drop

    // region:pointee
    #[lang = "pointee_trait"]
    pub trait Pointee: crate::core::marker::PointeeSized {
        #[lang = "metadata_type"]
        type Metadata: Copy + Send + Sync + Ord + Hash + Unpin;
    }

    #[lang = "dyn_metadata"]
    pub struct DynMetadata<Dyn: PointeeSized> {
        _phantom: crate::core::marker::PhantomData<Dyn>,
    }

    pub const fn metadata<T: PointeeSized>(ptr: *const T) -> <T as Pointee>::Metadata {
        loop {}
    }

    // endregion:pointee
    // region:non_null
    // `rustc_layout_scalar_valid_range_start` was removed in the 1.99 cycle (core
    // moved the null niche to pattern types). Keep it where it exists for the
    // layout optimization; dropping it on 1.99 leaves NonNull correct, just without
    // the `Option<NonNull>` niche. See build.rs `rustc_layout_range_attr_removed`.
    #[cfg_attr(not(rustc_layout_range_attr_removed), rustc_layout_scalar_valid_range_start(1))]
    #[rustc_nonnull_optimization_guaranteed]
    pub struct NonNull<T: crate::core::marker::PointeeSized> {
        pointer: *const T,
    }
    // region:coerce_unsized
    impl<T: crate::core::marker::PointeeSized, U: crate::core::marker::PointeeSized>
        crate::core::ops::CoerceUnsized<NonNull<U>> for NonNull<T>
    where
        T: crate::core::marker::Unsize<U>,
    {
    }
    // endregion:coerce_unsized
    // endregion:non_null

    pub const fn null<T>() -> *const T {
        0 as *const T
    }

    pub const fn null_mut<T>() -> *mut T {
        0 as *mut T
    }

    pub const fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
        unsafe { crate::core::mem::transmute((data, len)) }
    }

    pub const fn slice_from_raw_parts_mut<T>(data: *mut T, len: usize) -> *mut [T] {
        unsafe { crate::core::mem::transmute((data, len)) }
    }

    // region:addr_of
    #[rustc_macro_transparency = "semiopaque"]
    pub macro addr_of($place:expr) {
        &raw const $place
    }
    #[rustc_macro_transparency = "semiopaque"]
    pub macro addr_of_mut($place:expr) {
        &raw mut $place
    }
    // endregion:addr_of

    use crate::core::intrinsics;

    impl<T: PointeeSized> *mut T {
        #[rustc_allow_incoherent_impl]
        pub fn wrapping_add(self, count: usize) -> Self
        where
            T: Sized,
        {
            self.wrapping_offset(count as isize)
        }

        #[rustc_allow_incoherent_impl]
        pub fn wrapping_offset(self, count: isize) -> *mut T
        where
            T: Sized,
        {
            // SAFETY: the `arith_offset` intrinsic has no prerequisites to be called.
            unsafe { intrinsics::arith_offset(self, count) as *mut T }
        }
    }

    impl<T: PointeeSized> *const T {
        #[rustc_allow_incoherent_impl]
        pub fn wrapping_add(self, count: usize) -> Self
        where
            T: Sized,
        {
            self.wrapping_offset(count as isize)
        }

        #[rustc_allow_incoherent_impl]
        pub fn wrapping_offset(self, count: isize) -> *mut T
        where
            T: Sized,
        {
            // SAFETY: the `arith_offset` intrinsic has no prerequisites to be called.
            unsafe { intrinsics::arith_offset(self, count) as *mut T }
        }
    }
}

pub mod intrinsics {
    use crate::core::marker::DiscriminantKind;

    #[rustc_intrinsic]
    pub fn wrapping_add<T: Copy>(a: T, b: T) -> T;

    #[rustc_intrinsic]
    pub const unsafe fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;

    #[rustc_intrinsic]
    pub const unsafe fn unchecked_div<T: Copy>(x: T, y: T) -> T;

    #[rustc_intrinsic]
    pub const fn three_way_compare<T: Copy>(lhs: T, rhss: T) -> crate::core::cmp::Ordering;

    #[rustc_intrinsic]
    pub const fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;

    #[rustc_intrinsic]
    pub const fn assert_inhabited<T>();

    #[rustc_intrinsic]
    pub const unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);

    // Float intrinsics became `safe` right after 2025-08-04 (sqrt/exp/log stayed
    // non-const). Gate both spellings so tile_std builds on the pinned toolchain
    // and on current nightlies. See build.rs `rustc_float_intrinsics_safe`.
    // `expf32`/`logf32` became the generic `exp<T>`/`log<T>` on 2026-09 nightlies
    // (build.rs `rustc_float_math_generic`); nothing here calls them, so they are
    // simply not declared there.
    #[cfg(all(rustc_float_intrinsics_safe, not(rustc_float_math_generic)))]
    #[rustc_intrinsic]
    pub fn expf32(x: f32) -> f32;
    #[cfg(all(rustc_float_intrinsics_safe, not(rustc_float_math_generic)))]
    #[rustc_intrinsic]
    pub fn logf32(x: f32) -> f32;
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub fn sqrtf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn expf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn logf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn sqrtf32(x: f32) -> f32;

    // Bit manipulation intrinsics
    #[rustc_intrinsic]
    pub const fn ctpop<T: Copy>(x: T) -> u32;
    #[rustc_intrinsic]
    pub const fn ctlz<T: Copy>(x: T) -> u32;
    #[rustc_intrinsic]
    pub const fn cttz<T: Copy>(x: T) -> u32;
    #[rustc_intrinsic]
    pub const fn bswap<T: Copy>(x: T) -> T;
    #[rustc_intrinsic]
    pub const fn bitreverse<T: Copy>(x: T) -> T;
    #[rustc_intrinsic]
    pub const fn rotate_left<T: Copy>(x: T, shift: u32) -> T;
    #[rustc_intrinsic]
    pub const fn rotate_right<T: Copy>(x: T, shift: u32) -> T;

    // Float intrinsics — these became `safe const fn` right after 2025-08-04.
    // Gate both spellings (see build.rs `rustc_float_intrinsics_safe`).
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn floorf32(x: f32) -> f32;
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn ceilf32(x: f32) -> f32;
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn roundf32(x: f32) -> f32;
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn truncf32(x: f32) -> f32;
    // `fabsf32` intrinsic was removed by 2026-04-01 (and is unused here), so omit
    // it there; keep it on the safe-const nightlies that still provide it.
    // `fabsf32` and `copysignf32` became safe a day after the others (build.rs
    // `rustc_sign_intrinsics_safe`).
    #[cfg(all(rustc_sign_intrinsics_safe, not(rustc_fabsf32_removed)))]
    #[rustc_intrinsic]
    pub const fn fabsf32(x: f32) -> f32;
    #[cfg(rustc_sign_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn copysignf32(x: f32, y: f32) -> f32;
    #[cfg(rustc_float_intrinsics_safe)]
    #[rustc_intrinsic]
    pub const fn fmaf32(a: f32, b: f32, c: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn floorf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn ceilf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn roundf32(x: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn truncf32(x: f32) -> f32;
    #[cfg(not(rustc_sign_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn fabsf32(x: f32) -> f32;
    #[cfg(not(rustc_sign_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn copysignf32(x: f32, y: f32) -> f32;
    #[cfg(not(rustc_float_intrinsics_safe))]
    #[rustc_intrinsic]
    pub unsafe fn fmaf32(a: f32, b: f32, c: f32) -> f32;
    // `minnumf32`/`maxnumf32` intrinsics were removed after 2026-02-28 (superseded
    // by `minimumf32`/`maximumf32`, which differ in NaN/signed-zero semantics).
    // They were unused workspace-wide, so drop the dead declarations rather than
    // re-point them at intrinsics with different behavior.
}

pub mod ops {
    // region:coerce_unsized
    mod unsize {
        use crate::core::marker::{PointeeSized, Unsize};

        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a mut U> for &'a mut T {}
        impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b mut T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for &'a mut T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a mut T {}

        impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for *mut T {}
        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *mut T {}
        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *const T {}
    }
    pub use self::unsize::CoerceUnsized;
    // endregion:coerce_unsized

    // region:deref
    pub mod deref {
        use crate::core::marker::PointeeSized;

        #[lang = "deref"]
        pub trait Deref: PointeeSized {
            #[lang = "deref_target"]
            type Target: ?Sized;
            fn deref(&self) -> &Self::Target;
        }

        impl<T: ?Sized> Deref for &T {
            type Target = T;
            fn deref(&self) -> &T {
                *self
            }
        }
        impl<T: ?Sized> Deref for &mut T {
            type Target = T;
            fn deref(&self) -> &T {
                *self
            }
        }
        // region:deref_mut
        #[lang = "deref_mut"]
        pub trait DerefMut: Deref + PointeeSized {
            fn deref_mut(&mut self) -> &mut Self::Target;
        }
        // endregion:deref_mut

        // region:receiver
        #[lang = "receiver"]
        pub trait Receiver: PointeeSized {
            #[lang = "receiver_target"]
            type Target: ?Sized;
        }

        impl<P: PointeeSized, T: ?Sized> Receiver for P
        where
            P: Deref<Target = T>,
        {
            type Target = T;
        }

        #[lang = "legacy_receiver"]
        pub trait LegacyReceiver: PointeeSized {
            // Empty.
        }

        impl<T: PointeeSized> LegacyReceiver for &T {}

        impl<T: PointeeSized> LegacyReceiver for &mut T {}
        // endregion:receiver
    }
    pub use self::deref::{
        Deref,
        DerefMut, // :deref_mut
        LegacyReceiver,
        Receiver, // :receiver
    };
    // endregion:deref

    // region:drop
    #[lang = "drop"]
    pub trait Drop {
        fn drop(&mut self);
    }
    // endregion:drop

    // region:index
    pub mod index {
        #[lang = "index"]
        pub trait Index<Idx: ?Sized> {
            type Output: ?Sized;
            fn index(&self, index: Idx) -> &Self::Output;
        }
        #[lang = "index_mut"]
        pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
            fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
        }

        // region:slice
        impl<T> Index<usize> for [T] {
            type Output = T;
            fn index(&self, index: usize) -> &T {
                if index >= self.len() {
                    crate::core::panicking::panic("index out of bounds");
                }
                unsafe { &*self.as_ptr().wrapping_add(index) }
            }
        }
        impl<T> IndexMut<usize> for [T] {
            fn index_mut(&mut self, index: usize) -> &mut T {
                if index >= self.len() {
                    crate::core::panicking::panic("index out of bounds");
                }
                unsafe { &mut *self.as_mut_ptr().wrapping_add(index) }
            }
        }

        impl<T, const N: usize> Index<usize> for [T; N] {
            type Output = T;
            fn index(&self, index: usize) -> &T {
                let slice: &[T] = self;
                &slice[index]
            }
        }
        impl<T, const N: usize> IndexMut<usize> for [T; N] {
            fn index_mut(&mut self, index: usize) -> &mut T {
                let slice: &mut [T] = self;
                &mut slice[index]
            }
        }

        pub unsafe trait SliceIndex<T: ?Sized> {
            type Output: ?Sized;
        }
        unsafe impl<T> SliceIndex<[T]> for usize {
            type Output = T;
        }
        // endregion:slice
    }
    pub use self::index::{Index, IndexMut};
    // endregion:index

    // region:range
    pub mod range {
        #[lang = "RangeFull"]
        pub struct RangeFull;

        #[lang = "Range"]
        pub struct Range<Idx> {
            pub start: Idx,
            pub end: Idx,
        }

        #[lang = "RangeFrom"]
        pub struct RangeFrom<Idx> {
            pub start: Idx,
        }

        #[lang = "RangeTo"]
        pub struct RangeTo<Idx> {
            pub end: Idx,
        }

        #[lang = "RangeInclusive"]
        pub struct RangeInclusive<Idx> {
            pub(crate) start: Idx,
            pub(crate) end: Idx,
            pub(crate) exhausted: bool,
        }

        #[lang = "RangeToInclusive"]
        pub struct RangeToInclusive<Idx> {
            pub end: Idx,
        }
    }
    pub use self::range::{Range, RangeFrom, RangeFull, RangeTo};
    pub use self::range::{RangeInclusive, RangeToInclusive};
    // endregion:range

    // region:fn
    pub mod function {
        use crate::core::marker::Tuple;

        #[lang = "fn"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait Fn<Args: Tuple>: FnMut<Args> {
            extern "rust-call" fn call(&self, args: Args) -> Self::Output;
        }

        #[lang = "fn_mut"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait FnMut<Args: Tuple>: FnOnce<Args> {
            extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
        }

        #[lang = "fn_once"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait FnOnce<Args: Tuple> {
            #[lang = "fn_once_output"]
            type Output;
            extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
        }

        pub mod impls {
            use crate::core::marker::Tuple;
            use crate::marker::PointeeSized;

            impl<A: Tuple, F: ?Sized> Fn<A> for &F
            where
                F: Fn<A>,
            {
                extern "rust-call" fn call(&self, args: A) -> F::Output {
                    (**self).call(args)
                }
            }

            impl<A: Tuple, F: ?Sized> FnMut<A> for &F
            where
                F: Fn<A>,
            {
                extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
                    (**self).call(args)
                }
            }

            impl<A: Tuple, F: ?Sized> FnOnce<A> for &F
            where
                F: Fn<A>,
            {
                type Output = F::Output;

                extern "rust-call" fn call_once(self, args: A) -> F::Output {
                    (*self).call(args)
                }
            }

            impl<A: Tuple, F: ?Sized> FnMut<A> for &mut F
            where
                F: FnMut<A>,
            {
                extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
                    (*self).call_mut(args)
                }
            }

            impl<A: Tuple, F: ?Sized> FnOnce<A> for &mut F
            where
                F: FnMut<A>,
            {
                type Output = F::Output;
                extern "rust-call" fn call_once(self, args: A) -> F::Output {
                    (*self).call_mut(args)
                }
            }

            impl<A: PointeeSized, B: PointeeSized> PartialEq<&B> for &A
            where
                A: PartialEq<B>,
            {
                #[inline]
                fn eq(&self, other: &&B) -> bool {
                    PartialEq::eq(*self, *other)
                }
                #[inline]
                fn ne(&self, other: &&B) -> bool {
                    PartialEq::ne(*self, *other)
                }
            }
        }
    }
    pub use self::function::{Fn, FnMut, FnOnce};
    // endregion:fn

    pub use self::arith::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
    };
    pub use self::arith::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};
    pub use self::arith::{Shl, ShlAssign, Shr, ShrAssign};

    // region:async_fn
    pub mod async_function {
        use crate::core::{future::Future, marker::Tuple};

        #[lang = "async_fn"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFn<Args: Tuple>: AsyncFnMut<Args> {
            extern "rust-call" fn async_call(&self, args: Args) -> Self::CallRefFuture<'_>;
        }

        #[lang = "async_fn_mut"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFnMut<Args: Tuple>: AsyncFnOnce<Args> {
            #[lang = "call_ref_future"]
            type CallRefFuture<'a>: Future<Output = Self::Output>
            where
                Self: 'a;
            extern "rust-call" fn async_call_mut(&mut self, args: Args) -> Self::CallRefFuture<'_>;
        }

        #[lang = "async_fn_once"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFnOnce<Args: Tuple> {
            #[lang = "async_fn_once_output"]
            type Output;
            #[lang = "call_once_future"]
            type CallOnceFuture: Future<Output = Self::Output>;
            extern "rust-call" fn async_call_once(self, args: Args) -> Self::CallOnceFuture;
        }

        pub mod impls {
            use super::{AsyncFn, AsyncFnMut, AsyncFnOnce};
            use crate::core::marker::Tuple;

            impl<A: Tuple, F: ?Sized> AsyncFn<A> for &F
            where
                F: AsyncFn<A>,
            {
                extern "rust-call" fn async_call(&self, args: A) -> Self::CallRefFuture<'_> {
                    F::async_call(*self, args)
                }
            }

            impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &F
            where
                F: AsyncFn<A>,
            {
                type CallRefFuture<'a>
                    = F::CallRefFuture<'a>
                where
                    Self: 'a;

                extern "rust-call" fn async_call_mut(
                    &mut self,
                    args: A,
                ) -> Self::CallRefFuture<'_> {
                    F::async_call(*self, args)
                }
            }

            impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a F
            where
                F: AsyncFn<A>,
            {
                type Output = F::Output;
                type CallOnceFuture = F::CallRefFuture<'a>;

                extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
                    F::async_call(self, args)
                }
            }

            impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &mut F
            where
                F: AsyncFnMut<A>,
            {
                type CallRefFuture<'a>
                    = F::CallRefFuture<'a>
                where
                    Self: 'a;

                extern "rust-call" fn async_call_mut(
                    &mut self,
                    args: A,
                ) -> Self::CallRefFuture<'_> {
                    F::async_call_mut(*self, args)
                }
            }

            impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a mut F
            where
                F: AsyncFnMut<A>,
            {
                type Output = F::Output;
                type CallOnceFuture = F::CallRefFuture<'a>;

                extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
                    F::async_call_mut(self, args)
                }
            }
        }
    }
    pub use self::async_function::{AsyncFn, AsyncFnMut, AsyncFnOnce};
    // endregion:async_fn

    // region:try
    pub mod try_ {
        use crate::core::convert::Infallible;

        pub enum ControlFlow<B, C = ()> {
            #[lang = "Continue"]
            Continue(C),
            #[lang = "Break"]
            Break(B),
        }
        pub trait FromResidual<R = <Self as Try>::Residual> {
            #[lang = "from_residual"]
            fn from_residual(residual: R) -> Self;
        }
        #[lang = "Try"]
        pub trait Try: FromResidual<Self::Residual> {
            type Output;
            type Residual;
            #[lang = "from_output"]
            fn from_output(output: Self::Output) -> Self;
            #[lang = "branch"]
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
        }

        impl<B, C> Try for ControlFlow<B, C> {
            type Output = C;
            type Residual = ControlFlow<B, Infallible>;
            fn from_output(output: Self::Output) -> Self {
                ControlFlow::Continue(output)
            }
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    ControlFlow::Continue(x) => ControlFlow::Continue(x),
                    ControlFlow::Break(x) => ControlFlow::Break(ControlFlow::Break(x)),
                }
            }
        }

        impl<B, C> FromResidual for ControlFlow<B, C> {
            fn from_residual(residual: ControlFlow<B, Infallible>) -> Self {
                match residual {
                    ControlFlow::Break(b) => ControlFlow::Break(b),
                    ControlFlow::Continue(_) => loop {},
                }
            }
        }
        // region:option
        impl<T> Try for Option<T> {
            type Output = T;
            type Residual = Option<Infallible>;
            fn from_output(output: Self::Output) -> Self {
                Some(output)
            }
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    Some(x) => ControlFlow::Continue(x),
                    None => ControlFlow::Break(None),
                }
            }
        }

        impl<T> FromResidual for Option<T> {
            fn from_residual(x: Option<Infallible>) -> Self {
                match x {
                    None => None,
                    Some(_) => loop {},
                }
            }
        }
        // endregion:option
        // region:result
        // region:from
        use crate::core::convert::From;

        impl<T, E> Try for Result<T, E> {
            type Output = T;
            type Residual = Result<Infallible, E>;

            fn from_output(output: Self::Output) -> Self {
                Ok(output)
            }

            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    Ok(v) => ControlFlow::Continue(v),
                    Err(e) => ControlFlow::Break(Err(e)),
                }
            }
        }

        impl<T, E, F: From<E>> FromResidual<Result<Infallible, E>> for Result<T, F> {
            fn from_residual(residual: Result<Infallible, E>) -> Self {
                match residual {
                    Err(e) => Err(F::from(e)),
                    Ok(_) => loop {},
                }
            }
        }
        // endregion:from
        // endregion:result
    }
    pub use self::try_::{ControlFlow, FromResidual, Try};
    // endregion:try

    #[lang = "not"]
    pub trait Not {
        type Output;

        #[must_use]
        fn not(self) -> Self::Output;
    }

    macro_rules! not_impl {
        ($($t:ty)*) => ($(
            impl Not for $t {
                type Output = $t;

                #[inline]
                fn not(self) -> $t { !self }
            }
        )*)
    }

    not_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

    pub mod arith {
        #[lang = "neg"]
        pub trait Neg {
            type Output;

            fn neg(self) -> Self::Output;
        }

        macro_rules! neg_impl {
            ($($t:ty)*) => ($(
                impl Neg for $t {
                    type Output = $t;

                    #[inline]
                    #[rustc_inherit_overflow_checks]
                    fn neg(self) -> $t { -self }
                }
            )*)
        }

        neg_impl! { isize i8 i16 i32 i64 f16 f32 f128 }

        #[lang = "mul"]
        pub trait Mul<Rhs = Self> {
            type Output;
            fn mul(self, rhs: Rhs) -> Self::Output;
        }

        macro_rules! mul_impl {
            ($($t:ty)*) => ($(
                impl Mul for $t {
                    type Output = $t;

                    #[rustc_inherit_overflow_checks]
                    fn mul(self, other: $t) -> $t { self * other }
                }
            )*)
        }

        mul_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f16 f32 f128 }

        #[lang = "add"]
        pub trait Add<Rhs = Self> {
            type Output;
            fn add(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "add_assign"]
        pub trait AddAssign<Rhs = Self> {
            fn add_assign(&mut self, rhs: Rhs);
        }

        macro_rules! add_impl {
            ($($t:ty)*) => ($(
                impl Add for $t {
                    type Output = $t;
                    fn add(self, other: $t) -> $t { self + other }
                }
                impl AddAssign for $t {
                    fn add_assign(&mut self, other: $t) { *self += other; }
                }
            )*)
        }

        add_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f16 f32 f128 }

        #[lang = "div"]
        pub trait Div<Rhs = Self> {
            type Output;

            fn div(self, rhs: Rhs) -> Self::Output;
        }

        macro_rules! div_impl_integer {
            ($(($($t:ty)*) => $panic:expr),*) => ($($(
                impl Div for $t {
                    type Output = $t;

                    #[inline]
                    #[track_caller]
                    fn div(self, other: $t) -> $t { self / other }
                }
            )*)*)
        }

        macro_rules! div_impl_float {
            ($($t:ty)*) => ($(
                impl Div for $t {
                    type Output = $t;

                    #[inline]
                    fn div(self, other: $t) -> $t { self / other }
                }
            )*)
        }

        div_impl_float! { f16 f32 f128 }

        div_impl_integer! {
            (usize u8 u16 u32 u64) => "This operation will panic if `other == 0`.",
            (isize i8 i16 i32 i64) => "This operation will panic if `other == 0` or the division results in overflow."
        }

        // region:sub
        #[lang = "sub"]
        pub trait Sub<Rhs = Self> {
            type Output;
            fn sub(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "sub_assign"]
        pub trait SubAssign<Rhs = Self> {
            fn sub_assign(&mut self, rhs: Rhs);
        }

        macro_rules! sub_impl {
            ($($t:ty)*) => ($(
                impl Sub for $t {
                    type Output = $t;
                    #[rustc_inherit_overflow_checks]
                    fn sub(self, other: $t) -> $t { self - other }
                }
                impl SubAssign for $t {
                    fn sub_assign(&mut self, other: $t) { *self -= other; }
                }
            )*)
        }

        sub_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f16 f32 f128 }
        // endregion:sub

        // region:mul_assign
        #[lang = "mul_assign"]
        pub trait MulAssign<Rhs = Self> {
            fn mul_assign(&mut self, rhs: Rhs);
        }

        macro_rules! mul_assign_impl {
            ($($t:ty)*) => ($(
                impl MulAssign for $t {
                    fn mul_assign(&mut self, other: $t) { *self *= other; }
                }
            )*)
        }

        mul_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f16 f32 f128 }
        // endregion:mul_assign

        // region:div_assign
        #[lang = "div_assign"]
        pub trait DivAssign<Rhs = Self> {
            fn div_assign(&mut self, rhs: Rhs);
        }

        macro_rules! div_assign_impl {
            ($($t:ty)*) => ($(
                impl DivAssign for $t {
                    fn div_assign(&mut self, other: $t) { *self /= other; }
                }
            )*)
        }

        div_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f16 f32 f128 }
        // endregion:div_assign

        // region:rem
        #[lang = "rem"]
        pub trait Rem<Rhs = Self> {
            type Output;
            fn rem(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "rem_assign"]
        pub trait RemAssign<Rhs = Self> {
            fn rem_assign(&mut self, rhs: Rhs);
        }

        macro_rules! rem_impl_integer {
            ($(($($t:ty)*) => $panic:expr),*) => ($($(
                impl Rem for $t {
                    type Output = $t;

                    #[inline]
                    #[track_caller]
                    fn rem(self, other: $t) -> $t { self % other }
                }
                impl RemAssign for $t {
                    #[inline]
                    #[track_caller]
                    fn rem_assign(&mut self, other: $t) { *self %= other; }
                }
            )*)*)
        }

        macro_rules! rem_impl_float {
            ($($t:ty)*) => ($(
                impl Rem for $t {
                    type Output = $t;

                    #[inline]
                    fn rem(self, other: $t) -> $t { self % other }
                }
                impl RemAssign for $t {
                    #[inline]
                    fn rem_assign(&mut self, other: $t) { *self %= other; }
                }
            )*)
        }

        rem_impl_float! { f16 f32 f128 }

        rem_impl_integer! {
            (usize u8 u16 u32 u64) => "This operation will panic if `other == 0`.",
            (isize i8 i16 i32 i64) => "This operation will panic if `other == 0` or the division results in overflow."
        }
        // endregion:rem

        // region:bitand
        #[lang = "bitand"]
        pub trait BitAnd<Rhs = Self> {
            type Output;
            fn bitand(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "bitand_assign"]
        pub trait BitAndAssign<Rhs = Self> {
            fn bitand_assign(&mut self, rhs: Rhs);
        }

        macro_rules! bitand_impl {
            ($($t:ty)*) => ($(
                impl BitAnd for $t {
                    type Output = $t;
                    fn bitand(self, rhs: $t) -> $t { self & rhs }
                }
                impl BitAndAssign for $t {
                    fn bitand_assign(&mut self, other: $t) { *self &= other; }
                }
            )*)
        }

        bitand_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
        // endregion:bitand

        // region:bitor
        #[lang = "bitor"]
        pub trait BitOr<Rhs = Self> {
            type Output;
            fn bitor(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "bitor_assign"]
        pub trait BitOrAssign<Rhs = Self> {
            fn bitor_assign(&mut self, rhs: Rhs);
        }

        macro_rules! bitor_impl {
            ($($t:ty)*) => ($(
                impl BitOr for $t {
                    type Output = $t;
                    fn bitor(self, rhs: $t) -> $t { self | rhs }
                }
                impl BitOrAssign for $t {
                    fn bitor_assign(&mut self, other: $t) { *self |= other; }
                }
            )*)
        }

        bitor_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
        // endregion:bitor

        // region:bitxor
        #[lang = "bitxor"]
        pub trait BitXor<Rhs = Self> {
            type Output;
            fn bitxor(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "bitxor_assign"]
        pub trait BitXorAssign<Rhs = Self> {
            fn bitxor_assign(&mut self, rhs: Rhs);
        }

        macro_rules! bitxor_impl {
            ($($t:ty)*) => ($(
                impl BitXor for $t {
                    type Output = $t;
                    fn bitxor(self, rhs: $t) -> $t { self ^ rhs }
                }
                impl BitXorAssign for $t {
                    fn bitxor_assign(&mut self, other: $t) { *self ^= other; }
                }
            )*)
        }

        bitxor_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
        // endregion:bitxor

        // region:shl
        #[lang = "shl"]
        pub trait Shl<Rhs = Self> {
            type Output;
            fn shl(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "shl_assign"]
        pub trait ShlAssign<Rhs = Self> {
            fn shl_assign(&mut self, rhs: Rhs);
        }

        macro_rules! shl_impl {
            ($t:ty, $f:ty) => {
                impl Shl<$f> for $t {
                    type Output = $t;

                    #[rustc_inherit_overflow_checks]
                    fn shl(self, other: $f) -> $t {
                        self << other
                    }
                }
                impl ShlAssign<$f> for $t {
                    fn shl_assign(&mut self, other: $f) {
                        *self <<= other;
                    }
                }
            };
        }

        macro_rules! shl_impl_all {
            ($($t:ty)*) => ($(
                shl_impl! { $t, u8 }
                shl_impl! { $t, u16 }
                shl_impl! { $t, u32 }
                shl_impl! { $t, u64 }
                shl_impl! { $t, usize }
                shl_impl! { $t, i8 }
                shl_impl! { $t, i16 }
                shl_impl! { $t, i32 }
                shl_impl! { $t, i64 }
                shl_impl! { $t, isize }
            )*)
        }

        shl_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }
        // endregion:shl

        // region:shr
        #[lang = "shr"]
        pub trait Shr<Rhs = Self> {
            type Output;
            fn shr(self, rhs: Rhs) -> Self::Output;
        }

        #[lang = "shr_assign"]
        pub trait ShrAssign<Rhs = Self> {
            fn shr_assign(&mut self, rhs: Rhs);
        }

        macro_rules! shr_impl {
            ($t:ty, $f:ty) => {
                impl Shr<$f> for $t {
                    type Output = $t;

                    #[rustc_inherit_overflow_checks]
                    fn shr(self, other: $f) -> $t {
                        self >> other
                    }
                }
                impl ShrAssign<$f> for $t {
                    fn shr_assign(&mut self, other: $f) {
                        *self >>= other;
                    }
                }
            };
        }

        macro_rules! shr_impl_all {
            ($($t:ty)*) => ($(
                shr_impl! { $t, u8 }
                shr_impl! { $t, u16 }
                shr_impl! { $t, u32 }
                shr_impl! { $t, u64 }
                shr_impl! { $t, usize }
                shr_impl! { $t, i8 }
                shr_impl! { $t, i16 }
                shr_impl! { $t, i32 }
                shr_impl! { $t, i64 }
                shr_impl! { $t, isize }
            )*)
        }

        shr_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }
        // endregion:shr
    }

    // region:coroutine
    pub mod coroutine {
        use crate::core::pin::Pin;

        #[lang = "coroutine"]
        pub trait Coroutine<R = ()> {
            #[lang = "coroutine_yield"]
            type Yield;
            #[lang = "coroutine_return"]
            type Return;
            fn resume(self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return>;
        }

        #[lang = "coroutine_state"]
        pub enum CoroutineState<Y, R> {
            Yielded(Y),
            Complete(R),
        }
    }
    pub use self::coroutine::{Coroutine, CoroutineState};
    // endregion:coroutine

    // region:dispatch_from_dyn
    pub mod dispatch_from_dyn {
        use crate::core::marker::{PointeeSized, Unsize};

        #[lang = "dispatch_from_dyn"]
        pub trait DispatchFromDyn<T> {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a U> for &'a T {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a mut U> for &'a mut T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*const U> for *const T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*mut U> for *mut T {}
    }
    pub use self::dispatch_from_dyn::DispatchFromDyn;
    // endregion:dispatch_from_dyn
}

// region:eq
pub mod cmp {
    use crate::core::marker::PointeeSized;

    #[lang = "eq"]
    pub trait PartialEq<Rhs: PointeeSized = Self>: PointeeSized {
        fn eq(&self, other: &Rhs) -> bool;
        fn ne(&self, other: &Rhs) -> bool {
            !self.eq(other)
        }
    }

    pub trait Eq: PartialEq<Self> + PointeeSized {}

    macro_rules! partial_eq_impl {
        ($($t:ty)*) => ($(
            impl PartialEq for $t {
                fn eq(&self, other: &Self) -> bool {
                    *self == *other
                }
            }
        )*)
    }

    partial_eq_impl! {
        usize bool char u8 u16 u32 u64 i8 isize i16 i32 i64 f32
    }

    macro_rules! eq_impl {
        ($($t:ty)*) => ($(
            impl Eq for $t {}
        )*)
    }

    eq_impl! {
        usize bool char u8 u16 u32 u64 i8 isize i16 i32 i64
    }

    // region:builtin_impls
    impl PartialEq for () {
        fn eq(&self, other: &()) -> bool {
            true
        }
    }
    // endregion:builtin_impls

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialEq($item:item) {}
    #[rustc_builtin_macro]
    pub macro Eq($item:item) {}
    // endregion:derive

    // region:ord
    #[lang = "partial_ord"]
    pub trait PartialOrd<Rhs: PointeeSized = Self>: PartialEq<Rhs> + PointeeSized {
        fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;

        fn lt(&self, other: &Rhs) -> bool {
            self.partial_cmp(other).is_some_and(Ordering::is_lt)
        }

        fn le(&self, other: &Rhs) -> bool {
            self.partial_cmp(other).is_some_and(Ordering::is_le)
        }

        fn gt(&self, other: &Rhs) -> bool {
            self.partial_cmp(other).is_some_and(Ordering::is_gt)
        }

        fn ge(&self, other: &Rhs) -> bool {
            self.partial_cmp(other).is_some_and(Ordering::is_ge)
        }
    }

    pub trait Ord: Eq + PartialOrd<Self> + PointeeSized {
        fn cmp(&self, other: &Self) -> Ordering;
    }

    #[derive(PartialEq, PartialOrd)]
    #[lang = "Ordering"]
    #[repr(i8)]
    pub enum Ordering {
        Less = -1,
        Equal = 0,
        Greater = 1,
    }

    impl Ordering {
        const fn as_raw(self) -> i8 {
            crate::core::intrinsics::discriminant_value(&self)
        }

        pub const fn is_lt(self) -> bool {
            self.as_raw() < 0
        }

        pub const fn is_le(self) -> bool {
            self.as_raw() <= 0
        }

        pub const fn is_gt(self) -> bool {
            self.as_raw() > 0
        }

        pub const fn is_ge(self) -> bool {
            self.as_raw() >= 0
        }
    }

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialOrd($item:item) {}
    #[rustc_builtin_macro]
    pub macro Ord($item:item) {}
    // endregion:derive

    macro_rules! ord_impl {
        ($($t:ty)*) => ($(
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(crate::core::intrinsics::three_way_compare(*self, *other))
                }
            }

            impl Ord for $t {
                #[inline]
                fn cmp(&self, other: &Self) -> Ordering {
                    crate::core::intrinsics::three_way_compare(*self, *other)
                }
            }
        )*)
    }

    ord_impl! { char usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

    macro_rules! partial_ord_impl {
        ($($t:ty)*) => ($(
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    match (*self <= *other, *self >= *other) {
                        (false, false) => None,
                        (false, true) => Some(Greater),
                        (true, false) => Some(Less),
                        (true, true) => Some(Equal),
                    }
                }

                // partial_ord_methods_primitive_impl!();
            }
        )*)
    }

    impl PartialOrd for () {
        #[inline]
        fn partial_cmp(&self, _: &()) -> Option<Ordering> {
            Some(Equal)
        }
    }

    partial_ord_impl! { f32 /* f16 f128 */ }

    // endregion:ord
}
// endregion:eq

// region:fmt
pub mod fmt {
    use crate::core::marker::PointeeSized;

    pub struct Error;
    pub type Result = crate::core::result::Result<(), Error>;
    pub struct Formatter<'a>(&'a ());
    pub struct DebugTuple;
    pub struct DebugStruct;
    impl Formatter<'_> {
        pub fn debug_tuple(&mut self, _name: &str) -> DebugTuple {
            DebugTuple
        }

        pub fn debug_struct(&mut self, _name: &str) -> DebugStruct {
            DebugStruct
        }
    }

    impl DebugTuple {
        pub fn field(&mut self, _value: &dyn Debug) -> &mut Self {
            self
        }

        pub fn finish(&mut self) -> Result {
            Ok(())
        }
    }

    impl DebugStruct {
        pub fn field(&mut self, _name: &str, _value: &dyn Debug) -> &mut Self {
            self
        }

        pub fn finish(&mut self) -> Result {
            Ok(())
        }
    }

    pub trait Debug: PointeeSized {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }
    pub trait Display: PointeeSized {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }

    mod rt {
        use super::*;

        extern "C" {
            type Opaque;
        }

        #[derive(Copy, Clone)]
        #[lang = "format_argument"]
        pub struct Argument<'a> {
            value: &'a Opaque,
            formatter: fn(&Opaque, &mut Formatter<'_>) -> Result,
        }

        impl<'a> Argument<'a> {
            pub fn new<'b, T>(x: &'b T, f: fn(&T, &mut Formatter<'_>) -> Result) -> Argument<'b> {
                use crate::core::mem::transmute;
                unsafe {
                    Argument {
                        formatter: transmute(f),
                        value: transmute(x),
                    }
                }
            }

            pub fn new_display<'b, T: crate::core::fmt::Display>(x: &'b T) -> Argument<'b> {
                Self::new(x, crate::core::fmt::Display::fmt)
            }

            pub fn new_debug<T: Debug>(x: &T) -> Argument<'_> {
                Self::new(x, crate::core::fmt::Debug::fmt)
            }
        }

        // #[lang = "format_alignment"]
        pub enum Alignment {
            Left,
            Right,
            Center,
            Unknown,
        }

        // `format_count`/`format_placeholder`/`format_unsafe_arg` are no longer
        // lang items after 2025-08-04; the `format_args!` expansion resolves these
        // types by path. Keep the types, drop the removed `#[lang]` attributes.
        pub enum Count {
            Is(usize),
            Param(usize),
            Implied,
        }

        pub struct Placeholder {
            pub position: usize,
            pub fill: char,
            pub align: Alignment,
            pub flags: u32,
            pub precision: Count,
            pub width: Count,
        }

        impl Placeholder {
            pub const fn new(
                position: usize,
                fill: char,
                align: Alignment,
                flags: u32,
                precision: Count,
                width: Count,
            ) -> Self {
                Placeholder {
                    position,
                    fill,
                    align,
                    flags,
                    precision,
                    width,
                }
            }
        }

        // region:fmt_before_1_89_0
        pub struct UnsafeArg {
            _private: (),
        }

        impl UnsafeArg {
            pub unsafe fn new() -> Self {
                UnsafeArg { _private: () }
            }
        }
        // endregion:fmt_before_1_89_0
    }

    #[derive(Copy, Clone)]
    #[lang = "format_arguments"]
    pub struct Arguments<'a> {
        pieces: &'a [&'static str],
        fmt: Option<&'a [rt::Placeholder]>,
        args: &'a [rt::Argument<'a>],
    }

    impl<'a> Arguments<'a> {
        pub const fn new_v1(
            pieces: &'a [&'static str],
            args: &'a [rt::Argument<'a>],
        ) -> Arguments<'a> {
            Arguments {
                pieces,
                fmt: None,
                args,
            }
        }

        pub const fn new_const(pieces: &'a [&'static str]) -> Arguments<'a> {
            Arguments {
                pieces,
                fmt: None,
                args: &[],
            }
        }


        pub fn new_v1_formatted(
            pieces: &'a [&'static str],
            args: &'a [rt::Argument<'a>],
            fmt: &'a [rt::Placeholder],
            _unsafe_arg: rt::UnsafeArg,
        ) -> Arguments<'a> {
            Arguments {
                pieces,
                fmt: Some(fmt),
                args,
            }
        }

        pub const fn as_str(&self) -> Option<&'static str> {
            match (self.pieces, self.args) {
                ([], []) => Some(""),
                ([s], []) => Some(s),
                _ => None,
            }
        }
    }

    pub fn format(args: Arguments<'_>) -> &'static str {
        loop {}
    }

    // region:derive
    pub(crate) mod derive {
        #[rustc_builtin_macro]
        pub macro Debug($item:item) {}
    }
    pub use derive::Debug;
    // endregion:derive

    // region:builtin_impls
    macro_rules! impl_debug {
        ($($t:ty)*) => {
            $(
                impl Debug for $t {
                    fn fmt(&self, _f: &mut Formatter<'_>) -> Result {
                        Ok(())
                    }
                }
            )*
        }
    }

    impl_debug! {
        usize u8 u16 u32 u64
        isize i8 i16 i32 i64
        f16 f32 f128
        bool char
    }

    impl<T: Debug> Debug for [T] {
        fn fmt(&self, _f: &mut Formatter<'_>) -> Result {
            Ok(())
        }
    }

    impl<T: Debug + PointeeSized> Debug for &T {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            (&**self).fmt(f)
        }
    }

    impl Display for str {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            Ok(())
        }
    }
    // endregion:builtin_impls

    macro_rules! peel {
        ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
    }

    macro_rules! tuple {
        () => ();
        ( $($name:ident,)+ ) => (
            maybe_tuple_doc! {
                $($name)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($name:Debug),+> Debug for ($($name,)+) {
                    #[allow(non_snake_case, unused_assignments)]
                    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                        let mut builder = f.debug_tuple("");
                        let ($(ref $name,)+) = *self;
                        $(
                            builder.field(&$name);
                        )+

                        builder.finish()
                    }
                }
            }
            peel! { $($name,)+ }
        )
    }

    macro_rules! maybe_tuple_doc {
        ($a:ident @ #[$meta:meta] $item:item) => {
            #[doc = "This trait is implemented for tuples up to twelve items long."]
            $item
        };
        ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
            #[doc(hidden)]
            $item
        };
    }

    tuple! { E, D, C, B, A, Z, Y, X, W, V, U, T, }

    impl Debug for str {
        fn fmt(&self, _f: &mut Formatter<'_>) -> Result {
            Ok(())
        }
    }
}
// endregion:fmt

// region:slice
pub mod slice {
    // #[lang = "slice"]
    impl<T> [T] {
        #[lang = "slice_len_fn"]
        #[rustc_allow_incoherent_impl]
        pub const fn len(&self) -> usize {
            // Implemented by the compiler; unreachable at runtime for codegen.
            loop {}
        }

        #[rustc_allow_incoherent_impl]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        #[rustc_allow_incoherent_impl]
        pub fn first(&self) -> Option<&T> {
            if self.is_empty() {
                None
            } else {
                Some(&self[0])
            }
        }

        #[rustc_allow_incoherent_impl]
        pub fn last(&self) -> Option<&T> {
            if self.is_empty() {
                None
            } else {
                Some(&self[self.len() - 1])
            }
        }

        #[rustc_allow_incoherent_impl]
        pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
            if mid > self.len() {
                crate::core::panicking::panic("split_at: mid > len");
            }
            // SAFETY: `[ptr; mid]` and `[mid; len]` are inside `self`, which
            // fulfills the requirements of `split_at_unchecked`.
            let ptr = self.as_ptr();
            unsafe {
                let left = crate::core::slice::from_raw_parts(ptr, mid);
                let right =
                    crate::core::slice::from_raw_parts(ptr.wrapping_add(mid), self.len() - mid);
                (left, right)
            }
        }

        #[rustc_allow_incoherent_impl]
        pub fn contains(&self, x: &T) -> bool
        where
            T: PartialEq,
        {
            let mut i = 0;
            while i < self.len() {
                if self[i] == *x {
                    return true;
                }
                i += 1;
            }
            false
        }

        #[rustc_allow_incoherent_impl]
        pub fn reverse(&mut self) {
            let len = self.len();
            if len <= 1 {
                return;
            }
            let mut i = 0;
            let half = len / 2;
            while i < half {
                self.swap(i, len - 1 - i);
                i += 1;
            }
        }

        #[rustc_allow_incoherent_impl]
        pub fn swap(&mut self, a: usize, b: usize) {
            if a >= self.len() {
                crate::core::panicking::panic("swap: a out of bounds");
            }
            if b >= self.len() {
                crate::core::panicking::panic("swap: b out of bounds");
            }
            if a == b {
                return;
            }
            // Byte-by-byte swap avoids requiring T: Copy
            let pa = &mut self[a] as *mut T as *mut u8;
            let pb = &mut self[b] as *mut T as *mut u8;
            let size = crate::core::mem::size_of::<T>();
            unsafe {
                let mut i = 0usize;
                while i < size {
                    let a_byte = pa.wrapping_add(i);
                    let b_byte = pb.wrapping_add(i);
                    let tmp = *a_byte;
                    *a_byte = *b_byte;
                    *b_byte = tmp;
                    i = i + 1;
                }
            }
        }

        #[rustc_allow_incoherent_impl]
        pub fn iter(&self) -> crate::core::iter::Iter<'_, T> {
            // Delegates to IntoIterator
            self.into_iter()
        }

        #[rustc_allow_incoherent_impl]
        pub const fn as_ptr(&self) -> *const T {
            self as *const [T] as *const T
        }

        #[rustc_allow_incoherent_impl]
        pub fn as_mut_ptr(&mut self) -> *mut T {
            self as *mut [T] as *mut T
        }
    }

    pub unsafe fn from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
        unsafe { &*crate::core::ptr::slice_from_raw_parts(data, len) }
    }
}
// endregion:slice

// region:option
pub mod option {
    #[derive(Copy, Clone, PartialEq)]
    pub enum Option<T> {
        #[lang = "None"]
        None,
        #[lang = "Some"]
        Some(T),
    }

    impl<T> Option<T> {
        pub fn unwrap(self) -> T {
            match self {
                Some(val) => val,
                None => crate::panic!("called `Option::unwrap()` on a `None` value"),
            }
        }

        pub const fn as_ref(&self) -> Option<&T> {
            match self {
                Some(x) => Some(x),
                None => None,
            }
        }

        pub fn and<U>(self, optb: Option<U>) -> Option<U> {
            match self {
                Some(_) => optb,
                None => None,
            }
        }
        pub fn unwrap_or(self, default: T) -> T {
            match self {
                Some(val) => val,
                None => default,
            }
        }
        // region:result
        pub fn ok_or<E>(self, err: E) -> Result<T, E> {
            match self {
                Some(v) => Ok(v),
                None => Err(err),
            }
        }
        // endregion:result
        // region:fn
        pub fn and_then<U, F>(self, f: F) -> Option<U>
        where
            F: FnOnce(T) -> Option<U>,
        {
            match self {
                Some(x) => f(x),
                None => None,
            }
        }
        pub fn unwrap_or_else<F>(self, f: F) -> T
        where
            F: FnOnce() -> T,
        {
            match self {
                Some(val) => val,
                None => f(),
            }
        }
        pub fn map_or<U, F>(self, default: U, f: F) -> U
        where
            F: FnOnce(T) -> U,
        {
            match self {
                Some(x) => f(x),
                None => default,
            }
        }
        pub fn map_or_else<U, D, F>(self, default: D, f: F) -> U
        where
            D: FnOnce() -> U,
            F: FnOnce(T) -> U,
        {
            match self {
                Some(x) => f(x),
                None => default(),
            }
        }
        // endregion:fn

        pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> {
            match self {
                Some(x) => Some(f(x)),
                None => None,
            }
        }

        pub fn is_some(&self) -> bool {
            match self {
                Some(_) => true,
                None => false,
            }
        }

        pub fn is_none(&self) -> bool {
            !self.is_some()
        }

        pub fn is_some_and(self, f: impl FnOnce(T) -> bool) -> bool {
            match self {
                None => false,
                Some(x) => f(x),
            }
        }
    }
}
// endregion:option

// region:result
pub mod result {
    pub enum Result<T, E> {
        #[lang = "Ok"]
        Ok(T),
        #[lang = "Err"]
        Err(E),
    }

    impl<T, E> Result<T, E> {
        pub fn is_ok(&self) -> bool {
            match self {
                Ok(_) => true,
                Err(_) => false,
            }
        }

        pub fn is_err(&self) -> bool {
            !self.is_ok()
        }
    }
}
// endregion:result

// region:pin
pub mod pin {
    #[lang = "pin"]
    #[fundamental]
    pub struct Pin<P> {
        pointer: P,
    }
    impl<P> Pin<P> {
        pub fn new(pointer: P) -> Pin<P> {
            Pin { pointer }
        }
        pub unsafe fn new_unchecked(pointer: P) -> Pin<P> {
            Pin { pointer }
        }
    }
    // region:deref
    impl<P: crate::core::ops::Deref> crate::core::ops::Deref for Pin<P> {
        type Target = P::Target;
        fn deref(&self) -> &P::Target {
            &*self.pointer
        }
    }
    // endregion:deref
    // region:dispatch_from_dyn
    impl<Ptr, U> crate::core::ops::DispatchFromDyn<Pin<U>> for Pin<Ptr> where
        Ptr: crate::core::ops::DispatchFromDyn<U>
    {
    }
    // endregion:dispatch_from_dyn
    // region:coerce_unsized
    impl<Ptr, U> crate::core::ops::CoerceUnsized<Pin<U>> for Pin<Ptr> where
        Ptr: crate::core::ops::CoerceUnsized<U>
    {
    }
    // endregion:coerce_unsized

    use crate::ops::LegacyReceiver;
    impl<Ptr: LegacyReceiver> LegacyReceiver for Pin<Ptr> {}
}
// endregion:pin

// region:future
pub mod future {
    use crate::core::{
        pin::Pin,
        task::{Context, Poll},
    };

    #[doc(notable_trait)]
    #[lang = "future_trait"]
    pub trait Future {
        #[lang = "future_output"]
        type Output;
        #[lang = "poll"]
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
    }

    pub trait IntoFuture {
        type Output;
        type IntoFuture: Future<Output = Self::Output>;
        #[lang = "into_future"]
        fn into_future(self) -> Self::IntoFuture;
    }

    impl<F: Future> IntoFuture for F {
        type Output = F::Output;
        type IntoFuture = F;
        fn into_future(self) -> F {
            self
        }
    }
}
pub mod task {
    pub enum Poll<T> {
        #[lang = "Ready"]
        Ready(T),
        #[lang = "Pending"]
        Pending,
    }

    pub struct Context<'a> {
        pub waker: &'a (),
    }
}
// endregion:future

// region:iterator
pub mod iter {
    // region:iterators
    mod adapters {
        pub struct Take<I> {
            iter: I,
            n: usize,
        }
        impl<I> Iterator for Take<I>
        where
            I: Iterator,
        {
            type Item = <I as Iterator>::Item;

            fn next(&mut self) -> Option<<I as Iterator>::Item> {
                if self.n != 0 {
                    self.n -= 1;
                    self.iter.next()
                } else {
                    None
                }
            }
        }

        pub struct FilterMap<I, F> {
            iter: I,
            f: F,
        }
        impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
        where
            F: FnMut(I::Item) -> Option<B>,
        {
            type Item = B;

            #[inline]
            fn next(&mut self) -> Option<B> {
                while let Some(item) = self.iter.next() {
                    if let Some(mapped) = (self.f)(item) {
                        return Some(mapped);
                    }
                }
                None
            }
        }

        pub struct Map<I, F> {
            iter: I,
            f: F,
        }
        impl<B, I: Iterator, F> Iterator for Map<I, F>
        where
            F: FnMut(I::Item) -> B,
        {
            type Item = B;

            #[inline]
            fn next(&mut self) -> Option<B> {
                self.iter.next().map(|a| (self.f)(a))
            }
        }

        pub struct Filter<I, P> {
            iter: I,
            predicate: P,
        }
        impl<I: Iterator, P> Iterator for Filter<I, P>
        where
            P: FnMut(&I::Item) -> bool,
        {
            type Item = I::Item;

            #[inline]
            fn next(&mut self) -> Option<I::Item> {
                while let Some(item) = self.iter.next() {
                    if (self.predicate)(&item) {
                        return Some(item);
                    }
                }
                None
            }
        }

        pub struct Zip<A, B> {
            a: A,
            b: B,
        }
        impl<A: Iterator, B: Iterator> Iterator for Zip<A, B> {
            type Item = (A::Item, B::Item);

            #[inline]
            fn next(&mut self) -> Option<(A::Item, B::Item)> {
                let a = self.a.next()?;
                let b = self.b.next()?;
                Some((a, b))
            }
        }

        pub struct Enumerate<I> {
            iter: I,
            count: usize,
        }
        impl<I: Iterator> Iterator for Enumerate<I> {
            type Item = (usize, I::Item);

            #[inline]
            fn next(&mut self) -> Option<(usize, I::Item)> {
                let item = self.iter.next()?;
                let i = self.count;
                self.count += 1;
                Some((i, item))
            }
        }

        pub struct Chain<A, B> {
            a: Option<A>,
            b: B,
        }
        impl<A, B> Iterator for Chain<A, B>
        where
            A: Iterator,
            B: Iterator<Item = A::Item>,
        {
            type Item = A::Item;

            #[inline]
            fn next(&mut self) -> Option<A::Item> {
                match &mut self.a {
                    Some(a) => match a.next() {
                        item @ Some(_) => item,
                        None => {
                            self.a = None;
                            self.b.next()
                        }
                    },
                    None => self.b.next(),
                }
            }
        }

        impl<I> Take<I> {
            pub(crate) fn new(iter: I, n: usize) -> Take<I> {
                Take { iter, n }
            }
        }
        impl<I, F> FilterMap<I, F> {
            pub(crate) fn new(iter: I, f: F) -> FilterMap<I, F> {
                FilterMap { iter, f }
            }
        }
        impl<I, F> Map<I, F> {
            pub(crate) fn new(iter: I, f: F) -> Map<I, F> {
                Map { iter, f }
            }
        }
        impl<I, P> Filter<I, P> {
            pub(crate) fn new(iter: I, predicate: P) -> Filter<I, P> {
                Filter { iter, predicate }
            }
        }
        impl<A, B> Zip<A, B> {
            pub(crate) fn new(a: A, b: B) -> Zip<A, B> {
                Zip { a, b }
            }
        }
        impl<I> Enumerate<I> {
            pub(crate) fn new(iter: I) -> Enumerate<I> {
                Enumerate { iter, count: 0 }
            }
        }
        impl<A, B> Chain<A, B> {
            pub(crate) fn new(a: A, b: B) -> Chain<A, B> {
                Chain { a: Some(a), b }
            }
        }
    }
    pub use self::adapters::{Chain, Enumerate, Filter, FilterMap, Map, Take, Zip};

    mod sources {
        mod repeat {
            pub fn repeat<T>(elt: T) -> Repeat<T> {
                Repeat { element: elt }
            }

            pub struct Repeat<A> {
                element: A,
            }

            impl<A: Copy> Iterator for Repeat<A> {
                type Item = A;

                fn next(&mut self) -> Option<A> {
                    Some(self.element)
                }
            }
        }
        pub use self::repeat::{repeat, Repeat};
    }
    pub use self::sources::{repeat, Repeat};
    // endregion:iterators

    mod traits {
        mod iterator {
            use super::accum::{Product, Sum};
            use crate::core::marker::PointeeSized;

            #[doc(notable_trait)]
            #[lang = "iterator"]
            pub trait Iterator {
                type Item;
                #[lang = "next"]
                fn next(&mut self) -> Option<Self::Item>;
                fn nth(&mut self, n: usize) -> Option<Self::Item> {
                    let mut i = 0usize;
                    loop {
                        if i >= n {
                            return self.next();
                        }
                        if self.next().is_none() {
                            return None;
                        }
                        i = i + 1;
                    }
                }
                fn by_ref(&mut self) -> &mut Self
                where
                    Self: Sized,
                {
                    self
                }
                // region:iterators
                fn take(self, n: usize) -> crate::core::iter::Take<Self>
                where
                    Self: Sized,
                {
                    crate::core::iter::Take::new(self, n)
                }
                fn filter_map<B, F>(self, f: F) -> crate::core::iter::FilterMap<Self, F>
                where
                    Self: Sized,
                    F: FnMut(Self::Item) -> Option<B>,
                {
                    crate::core::iter::FilterMap::new(self, f)
                }
                fn map<B, F>(self, f: F) -> crate::core::iter::Map<Self, F>
                where
                    Self: Sized,
                    F: FnMut(Self::Item) -> B,
                {
                    crate::core::iter::Map::new(self, f)
                }
                fn filter<P>(self, predicate: P) -> crate::core::iter::Filter<Self, P>
                where
                    Self: Sized,
                    P: FnMut(&Self::Item) -> bool,
                {
                    crate::core::iter::Filter::new(self, predicate)
                }
                fn zip<U: IntoIterator>(self, other: U) -> crate::core::iter::Zip<Self, U::IntoIter>
                where
                    Self: Sized,
                {
                    crate::core::iter::Zip::new(self, other.into_iter())
                }
                fn enumerate(self) -> crate::core::iter::Enumerate<Self>
                where
                    Self: Sized,
                {
                    crate::core::iter::Enumerate::new(self)
                }
                fn chain<U: IntoIterator<Item = Self::Item>>(
                    self,
                    other: U,
                ) -> crate::core::iter::Chain<Self, U::IntoIter>
                where
                    Self: Sized,
                {
                    crate::core::iter::Chain::new(self, other.into_iter())
                }
                fn fold<B, F>(self, init: B, mut f: F) -> B
                where
                    Self: Sized,
                    F: FnMut(B, Self::Item) -> B,
                {
                    let mut accum = init;
                    let mut iter = self;
                    while let Some(x) = iter.next() {
                        accum = f(accum, x);
                    }
                    accum
                }
                fn sum<S: Sum<Self::Item>>(self) -> S
                where
                    Self: Sized,
                {
                    Sum::sum(self)
                }
                fn product<P: Product<Self::Item>>(self) -> P
                where
                    Self: Sized,
                {
                    Product::product(self)
                }
                fn collect<B: FromIterator<Self::Item>>(self) -> B
                where
                    Self: Sized,
                {
                    FromIterator::from_iter(self)
                }
                // endregion:iterators
            }
            impl<I: Iterator + PointeeSized> Iterator for &mut I {
                type Item = I::Item;
                fn next(&mut self) -> Option<I::Item> {
                    (**self).next()
                }
            }
        }
        pub use self::iterator::Iterator;

        mod collect {
            pub trait IntoIterator {
                type Item;
                type IntoIter: Iterator<Item = Self::Item>;
                #[lang = "into_iter"]
                fn into_iter(self) -> Self::IntoIter;
            }
            impl<I: Iterator> IntoIterator for I {
                type Item = I::Item;
                type IntoIter = I;
                fn into_iter(self) -> I {
                    self
                }
            }
            struct IndexRange {
                start: usize,
                end: usize,
            }
            pub struct IntoIter<T, const N: usize> {
                data: [T; N],
                range: IndexRange,
            }
            impl<T: Copy, const N: usize> IntoIterator for [T; N] {
                type Item = T;
                type IntoIter = IntoIter<T, N>;
                fn into_iter(self) -> Self::IntoIter {
                    IntoIter {
                        data: self,
                        range: IndexRange { start: 0, end: N },
                    }
                }
            }
            impl<T: Copy, const N: usize> Iterator for IntoIter<T, N> {
                type Item = T;
                fn next(&mut self) -> Option<T> {
                    if self.range.start < self.range.end {
                        let item = self.data[self.range.start];
                        self.range.start = self.range.start + 1;
                        Some(item)
                    } else {
                        None
                    }
                }
            }
            pub struct Iter<'a, T> {
                slice: &'a [T],
                pos: usize,
            }
            impl<'a, T, const N: usize> IntoIterator for &'a [T; N] {
                type Item = &'a T;
                type IntoIter = Iter<'a, T>;
                fn into_iter(self) -> Self::IntoIter {
                    Iter {
                        slice: self as &[T],
                        pos: 0,
                    }
                }
            }
            impl<'a, T> IntoIterator for &'a [T] {
                type Item = &'a T;
                type IntoIter = Iter<'a, T>;
                fn into_iter(self) -> Self::IntoIter {
                    Iter {
                        slice: self,
                        pos: 0,
                    }
                }
            }
            impl<'a, T> Iterator for Iter<'a, T> {
                type Item = &'a T;
                fn next(&mut self) -> Option<Self::Item> {
                    if self.pos < self.slice.len() {
                        let item = &self.slice[self.pos];
                        self.pos = self.pos + 1;
                        Some(item)
                    } else {
                        None
                    }
                }
            }
            pub trait FromIterator<A>: Sized {
                fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
            }
        }
        pub use self::collect::{FromIterator, IntoIterator, Iter};

        mod accum {
            pub trait Sum<A = Self>: Sized {
                fn sum<I: Iterator<Item = A>>(iter: I) -> Self;
            }

            pub trait Product<A = Self>: Sized {
                fn product<I: Iterator<Item = A>>(iter: I) -> Self;
            }

            macro_rules! integer_sum_product {
                (@sum $zero:expr; $($t:ty)*) => ($(
                    impl Sum for $t {
                        fn sum<I: Iterator<Item = $t>>(iter: I) -> $t {
                            iter.fold($zero, |a, b| a + b)
                        }
                    }
                )*);
                (@product $one:expr; $($t:ty)*) => ($(
                    impl Product for $t {
                        fn product<I: Iterator<Item = $t>>(iter: I) -> $t {
                            iter.fold($one, |a, b| a * b)
                        }
                    }
                )*);
            }

            integer_sum_product! { @sum 0; usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
            integer_sum_product! { @sum 0.0; f32 }
            integer_sum_product! { @product 1; usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
            integer_sum_product! { @product 1.0; f32 }
        }
        pub use self::accum::{Product, Sum};
    }
    pub use self::traits::{FromIterator, IntoIterator, Iter, Iterator, Product, Sum};
}
// endregion:iterator

// region:str
pub mod str {
    pub const unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
        unsafe { crate::core::mem::transmute(v) }
    }
    pub trait FromStr: Sized {
        type Err;
        fn from_str(s: &str) -> Result<Self, Self::Err>;
    }
    impl str {
        #[rustc_allow_incoherent_impl]
        pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
            FromStr::from_str(self)
        }

        #[rustc_allow_incoherent_impl]
        pub const fn len(&self) -> usize {
            self.as_bytes().len()
        }

        #[rustc_allow_incoherent_impl]
        pub const fn is_empty(&self) -> bool {
            self.len() == 0
        }

        #[rustc_allow_incoherent_impl]
        pub const fn as_bytes(&self) -> &[u8] {
            unsafe { crate::core::mem::transmute(self) }
        }

        #[rustc_allow_incoherent_impl]
        pub const fn as_ptr(&self) -> *const u8 {
            self as *const str as *const u8
        }
    }

    impl PartialEq for str {
        #[inline]
        fn eq(&self, other: &str) -> bool {
            if self.len() != other.len() {
                return false;
            }
            let left = self.as_bytes();
            let right = other.as_bytes();
            let mut i = 0;
            while i < left.len() {
                if left[i] != right[i] {
                    return false;
                }
                i += 1;
            }
            true
        }
    }

    impl Eq for str {}
}
// endregion:str

// region:panic
pub mod panic {
    pub macro panic_2021 {
        () => ({
            const fn panic_cold_explicit() -> ! {
                $crate::core::panicking::panic_explicit()
            }
            panic_cold_explicit();
        }),
        // Special-case the single-argument case for const_panic.
        ("{}", $arg:expr $(,)?) => ({
            #[rustc_const_panic_str] // enforce a &&str argument in const-check and hook this by const-eval
            #[rustc_do_not_const_check] // hooked by const-eval
            const fn panic_cold_display<T: $crate::core::fmt::Display>(arg: &T) -> ! {
                $crate::core::panicking::panic_display(arg)
            }
            panic_cold_display(&$arg);
        }),
        ($($t:tt)+) => ({
            // ANALYSIS-ONLY DIVERGENCE (nightly >2025-08-04 port): the built-in
            // `const_format_args!` expansion now emits `Arguments::from_str`/`new`
            // against a repacked `fmt::Arguments` layout this crate does not
            // implement. Since `panic_fmt` is a `loop {}` stub on-device and kernels
            // never format, route the formatted-panic arm through the surviving
            // `new_const` constructor with a placeholder instead of building real
            // format args. Restore `const_format_args!` if host-side panic messages
            // are needed (requires porting the `Arguments` ABI).
            $crate::core::panicking::panic_fmt(
                $crate::core::fmt::Arguments::new_const(&["explicit panic"]),
            );
        }),
    }

    #[lang = "panic_location"]
    struct PanicLocation {
        file: &'static str,
        line: u32,
        column: u32,
    }
}

pub mod panicking {
    // `#[rustc_const_panic_str]` was removed from rustc after nightly-2025-08-04;
    // it only enforced a `&&str` argument during const-check, so dropping it is safe here.
    pub fn panic_display<T: crate::core::fmt::Display>(x: &T) -> ! {
        // ANALYSIS-ONLY DIVERGENCE (nightly >2025-08-04 port): the built-in
        // `format_args!` expansion now targets a repacked `fmt::Arguments`
        // (`Arguments::from_str`/`new` over a NonNull tagged-pointer layout) that
        // this crate's pre-restructure `Arguments` does not implement. On-device
        // panics are `loop {}` stubs and kernels never format, so we diverge here
        // instead of porting the whole `Arguments` ABI. Restore real formatting if
        // host-side panic messages are ever needed. See panic_2021 catch-all below.
        let _ = x;
        loop {}
    }

    // This function is used instead of panic_fmt in const eval.
    #[lang = "const_panic_fmt"]
    pub fn const_panic_fmt(fmt: crate::core::fmt::Arguments<'_>) -> ! {
        // if let Some(msg) = fmt.as_str() {
        //     // The panic_display function is hooked by const eval.
        //     panic_display(msg);
        // } else {
        //     loop {}
        // }
        loop {}
    }

    #[lang = "panic_fmt"] // needed for const-evaluated panics
    pub fn panic_fmt(fmt: crate::core::fmt::Arguments<'_>) -> ! {
        loop {}
    }

    #[lang = "panic"]
    pub fn panic(expr: &'static str) -> ! {
        panic_fmt(crate::core::fmt::Arguments::new_const(&[expr]))
    }

    #[lang = "panic_cannot_unwind"]
    pub fn panic_cannot_unwind() -> ! {
        panic("panic in a function that cannot unwind")
    }

    macro_rules! panic_const {
        ($($lang:ident = $message:expr,)+) => {
            pub mod panic_const {
                use super::*;

                $(
                    #[track_caller]
                    #[lang = crate::core::macros::stringify!($lang)]
                    pub fn $lang() -> ! {
                        panic($message);
                    }
                )+
            }
        }
    }

    panic_const! {
        panic_const_add_overflow = "attempt to add with overflow",
        panic_const_sub_overflow = "attempt to subtract with overflow",
        panic_const_mul_overflow = "attempt to multiply with overflow",
        panic_const_div_overflow = "attempt to divide with overflow",
        panic_const_rem_overflow = "attempt to calculate the remainder with overflow",
        panic_const_neg_overflow = "attempt to negate with overflow",
        panic_const_shr_overflow = "attempt to shift right with overflow",
        panic_const_shl_overflow = "attempt to shift left with overflow",
        panic_const_div_by_zero = "attempt to divide by zero",
        panic_const_rem_by_zero = "attempt to calculate the remainder with a divisor of zero",
    }
}
// endregion:panic

// region:asm
mod arch {
    #[rustc_builtin_macro]
    pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
        /* compiler built-in */
    }
    #[rustc_builtin_macro]
    pub macro global_asm("assembly template", $(operands,)* $(options($(option),*))?) {
        /* compiler built-in */
    }
}
// endregion:asm

#[macro_use]
pub mod macros {
    #[rustc_builtin_macro]
    #[rustc_macro_transparency = "semiopaque"]
    pub macro stringify($($t:tt)*) {
        /* compiler built-in */
    }

    // region:panic
    #[macro_export]
    #[rustc_builtin_macro(core_panic)]
    macro_rules! panic {
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }
    // endregion:panic

    // region:write
    #[macro_export]
    macro_rules! write {
        ($dst:expr, $($arg:tt)*) => {
            $dst.write_fmt($crate::format_args!($($arg)*))
        };
    }

    #[macro_export]
    #[allow_internal_unstable(format_args_nl)]
    macro_rules! writeln {
        ($dst:expr $(,)?) => {
            $crate::core::write!($dst, "\n")
        };
        ($dst:expr, $($arg:tt)*) => {
            $dst.write_fmt($crate::format_args_nl!($($arg)*))
        };
    }
    // endregion:write

    // region:assert
    #[macro_export]
    #[rustc_builtin_macro]
    #[allow_internal_unstable(core_panic, edition_panic, generic_assert_internals)]
    macro_rules! assert {
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }

    #[macro_export]
    macro_rules! assert_eq {
        ($l:expr, $r: expr) => {
            if $l != $r {
                $crate::core::panicking::panic(crate::core::macros::stringify!($l != $r));
            }
        };
    }

    #[macro_export]
    macro_rules! assert_ne {
        ($a:expr, $b:expr) => {
            let _a = $a;
            let _b = $b;

            if _a == _b { /* todo */ }
        };
    }

    // endregion:assert

    #[macro_export]
    #[rustc_builtin_macro(unreachable)]
    #[allow_internal_unstable(edition_panic)]
    macro_rules! unreachable {
        // Expands to either `$crate::panic::unreachable_2015` or `$crate::panic::unreachable_2021`
        // depending on the edition of the caller.
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }

    // region:fmt
    #[allow_internal_unstable(fmt_internals, const_fmt_arguments_new)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! const_format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[allow_internal_unstable(fmt_internals)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[macro_export]
    macro_rules! format {
        ($($arg:tt)*) => {
            $crate::fmt::format($crate::format_args!($($arg)*))
        }
    }

    #[allow_internal_unstable(fmt_internals)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! format_args_nl {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[macro_export]
    macro_rules! print {
        ($($arg:tt)*) => {{
            $crate::core::io::_print($crate::format_args!($($arg)*));
        }};
    }

    // endregion:fmt

    // region:todo
    #[macro_export]
    #[allow_internal_unstable(core_panic)]
    macro_rules! todo {
        () => {
            $crate::core::panicking::panic("not yet implemented")
        };
        ($($arg:tt)+) => {
            $crate::core::panic!("not yet implemented: {}", $crate::format_args!($($arg)+))
        };
    }
    // endregion:todo

    // region:unimplemented
    #[macro_export]
    #[allow_internal_unstable(core_panic)]
    macro_rules! unimplemented {
        () => {
            $crate::core::panicking::panic("not implemented")
        };
        ($($arg:tt)+) => {
            $crate::core::panic!("not implemented: {}", $crate::format_args!($($arg)+))
        };
    }
    // endregion:unimplemented

    // region:derive
    pub mod builtin {
        #[rustc_builtin_macro]
        pub macro derive($item:item) {
            /* compiler built-in */
        }

        #[rustc_builtin_macro]
        pub macro derive_const($item:item) {
            /* compiler built-in */
        }
    }
    // endregion:derive

    // region:include
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! include {
        ($file:expr $(,)?) => {{ /* compiler built-in */ }};
    }
    // endregion:include

    // region:concat
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat {
        () => {};
    }
    // endregion:concat

    // region:env
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! env {
        () => {};
    }
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! option_env {
        () => {};
    }
    // endregion:env
}

// region:non_zero
pub mod num {
    #[repr(transparent)]
    // Removed in 1.99 (see NonNull above / build.rs `rustc_layout_range_attr_removed`).
    #[cfg_attr(not(rustc_layout_range_attr_removed), rustc_layout_scalar_valid_range_start(1))]
    #[rustc_nonnull_optimization_guaranteed]
    pub struct NonZeroU8(u8);

    pub mod f64 {
        // f64 is not natively supported on NPU, but this module is needed
        // for lang item completeness. Avoid using f64 in kernel code.
    }
}
// endregion:non_zero

// region:bool_impl
// #[lang = "bool"]
impl bool {
    #[rustc_allow_incoherent_impl]
    pub fn then_some<T>(self, t: T) -> Option<T> {
        if self {
            Some(t)
        } else {
            None
        }
    }

    #[rustc_allow_incoherent_impl]
    pub fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self {
            Some(f())
        } else {
            None
        }
    }
}
// endregion:bool_impl

// region:int_impl
macro_rules! impl_int {
    ($($t:ty)*) => {
        $(
            impl $t {
                #[rustc_allow_incoherent_impl]
                pub fn from_ne_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
                    // Cannot use mem::transmute here — triggers ICE on nightly-2025-08-04
                    // with const generic expressions in no_core context.
                    // Use ptr::read via pointer cast instead.
                    unsafe { crate::core::ptr::read(&bytes as *const _ as *const Self) }
                }
            }
        )*
    }
}

impl_int! {
    usize u8 u16 u32 u64
    isize i8 i16 i32 i64
}

macro_rules! impl_uint_methods {
    ($($t:ty)*) => ($(
        impl $t {
            pub const MIN: $t = 0;
            pub const MAX: $t = !0;

            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_add(self, rhs: $t) -> $t {
                // intrinsic wrapping is the default for Rust arithmetic in release
                self + rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_sub(self, rhs: $t) -> $t {
                self - rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_mul(self, rhs: $t) -> $t {
                self * rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn saturating_add(self, rhs: $t) -> $t {
                let (result, overflow) = (self + rhs, self > <$t>::MAX - rhs);
                if overflow { <$t>::MAX } else { result }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn saturating_sub(self, rhs: $t) -> $t {
                if self < rhs { 0 } else { self - rhs }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn overflowing_add(self, rhs: $t) -> ($t, bool) {
                let result = self.wrapping_add(rhs);
                (result, self > <$t>::MAX - rhs)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn overflowing_sub(self, rhs: $t) -> ($t, bool) {
                let result = self.wrapping_sub(rhs);
                (result, self < rhs)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_add(self, rhs: $t) -> Option<$t> {
                if self > <$t>::MAX - rhs { None } else { Some(self + rhs) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_sub(self, rhs: $t) -> Option<$t> {
                if self < rhs { None } else { Some(self - rhs) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_mul(self, rhs: $t) -> Option<$t> {
                // NPU: skip overflow detection to avoid LLVM widening u64*u64
                // to i128 (__multi3). The division-based check triggers LLVM's
                // pattern recognizer regardless of how the multiply is written.
                Some(self * rhs)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn min(self, other: $t) -> $t {
                if self < other { self } else { other }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn max(self, other: $t) -> $t {
                if self > other { self } else { other }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn pow(self, mut exp: u32) -> $t {
                let mut base = self;
                let mut acc: $t = 1;
                while exp > 1 {
                    if (exp & 1) == 1 {
                        acc = acc * base;
                    }
                    exp /= 2;
                    base = base * base;
                }
                if exp == 1 {
                    acc = acc * base;
                }
                acc
            }

            #[rustc_allow_incoherent_impl]
            pub const fn count_ones(self) -> u32 {
                crate::core::intrinsics::ctpop(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn count_zeros(self) -> u32 {
                (!self).count_ones()
            }
            #[rustc_allow_incoherent_impl]
            pub const fn leading_zeros(self) -> u32 {
                crate::core::intrinsics::ctlz(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn trailing_zeros(self) -> u32 {
                crate::core::intrinsics::cttz(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn swap_bytes(self) -> $t {
                crate::core::intrinsics::bswap(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn reverse_bits(self) -> $t {
                crate::core::intrinsics::bitreverse(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn rotate_left(self, n: u32) -> $t {
                crate::core::intrinsics::rotate_left(self, n)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn rotate_right(self, n: u32) -> $t {
                crate::core::intrinsics::rotate_right(self, n)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn is_power_of_two(self) -> bool {
                self > 0 && (self & (self - 1)) == 0
            }
            #[rustc_allow_incoherent_impl]
            #[allow(unnecessary_transmutes)]
            pub const fn to_ne_bytes(self) -> [u8; size_of::<Self>()] {
                // Safety: any bit pattern is valid for [u8; N]
                unsafe { crate::core::mem::transmute(self) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn to_be(self) -> $t {
                #[cfg(target_endian = "big")]
                { self }
                #[cfg(not(target_endian = "big"))]
                { self.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn to_le(self) -> $t {
                #[cfg(target_endian = "little")]
                { self }
                #[cfg(not(target_endian = "little"))]
                { self.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn from_be(x: $t) -> $t {
                #[cfg(target_endian = "big")]
                { x }
                #[cfg(not(target_endian = "big"))]
                { x.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn from_le(x: $t) -> $t {
                #[cfg(target_endian = "little")]
                { x }
                #[cfg(not(target_endian = "little"))]
                { x.swap_bytes() }
            }
        }
    )*)
}

impl_uint_methods! { usize u8 u16 u32 u64 }

macro_rules! impl_sint_methods {
    ($($t:ty, $ut:ty);*) => ($(
        impl $t {
            pub const MIN: $t = !(<$t>::MAX);
            pub const MAX: $t = (!0 as $ut >> 1) as $t;

            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_add(self, rhs: $t) -> $t {
                self + rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_sub(self, rhs: $t) -> $t {
                self - rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_mul(self, rhs: $t) -> $t {
                self * rhs
            }
            #[rustc_allow_incoherent_impl]
            pub const fn wrapping_neg(self) -> $t {
                (0 as $t).wrapping_sub(self)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn abs(self) -> $t {
                if self < 0 { -self } else { self }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn saturating_add(self, rhs: $t) -> $t {
                let result = self.wrapping_add(rhs);
                // Overflow if signs of operands are same but result sign differs
                if (rhs >= 0 && result < self) {
                    <$t>::MAX
                } else if (rhs < 0 && result > self) {
                    <$t>::MIN
                } else {
                    result
                }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn saturating_sub(self, rhs: $t) -> $t {
                let result = self.wrapping_sub(rhs);
                if (rhs > 0 && result > self) {
                    <$t>::MIN
                } else if (rhs < 0 && result < self) {
                    <$t>::MAX
                } else {
                    result
                }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn overflowing_add(self, rhs: $t) -> ($t, bool) {
                let result = self.wrapping_add(rhs);
                let overflow = (rhs >= 0 && result < self) || (rhs < 0 && result > self);
                (result, overflow)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn overflowing_sub(self, rhs: $t) -> ($t, bool) {
                let result = self.wrapping_sub(rhs);
                let overflow = (rhs > 0 && result > self) || (rhs < 0 && result < self);
                (result, overflow)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_add(self, rhs: $t) -> Option<$t> {
                let (result, overflow) = self.overflowing_add(rhs);
                if overflow { None } else { Some(result) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_sub(self, rhs: $t) -> Option<$t> {
                let (result, overflow) = self.overflowing_sub(rhs);
                if overflow { None } else { Some(result) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn checked_mul(self, rhs: $t) -> Option<$t> {
                // NPU: skip overflow detection (same as unsigned — avoid i128 widening)
                Some(self * rhs)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn min(self, other: $t) -> $t {
                if self < other { self } else { other }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn max(self, other: $t) -> $t {
                if self > other { self } else { other }
            }

            #[rustc_allow_incoherent_impl]
            pub const fn count_ones(self) -> u32 {
                crate::core::intrinsics::ctpop(self as $ut)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn count_zeros(self) -> u32 {
                (!self).count_ones()
            }
            #[rustc_allow_incoherent_impl]
            pub const fn leading_zeros(self) -> u32 {
                (self as $ut).leading_zeros()
            }
            #[rustc_allow_incoherent_impl]
            pub const fn trailing_zeros(self) -> u32 {
                (self as $ut).trailing_zeros()
            }
            #[rustc_allow_incoherent_impl]
            pub const fn swap_bytes(self) -> $t {
                (self as $ut).swap_bytes() as $t
            }
            #[rustc_allow_incoherent_impl]
            pub const fn reverse_bits(self) -> $t {
                (self as $ut).reverse_bits() as $t
            }
            #[rustc_allow_incoherent_impl]
            pub const fn rotate_left(self, n: u32) -> $t {
                (self as $ut).rotate_left(n) as $t
            }
            #[rustc_allow_incoherent_impl]
            pub const fn rotate_right(self, n: u32) -> $t {
                (self as $ut).rotate_right(n) as $t
            }
            #[rustc_allow_incoherent_impl]
            #[allow(unnecessary_transmutes)]
            pub const fn to_ne_bytes(self) -> [u8; size_of::<Self>()] {
                unsafe { crate::core::mem::transmute(self) }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn to_be(self) -> $t {
                #[cfg(target_endian = "big")]
                { self }
                #[cfg(not(target_endian = "big"))]
                { self.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn to_le(self) -> $t {
                #[cfg(target_endian = "little")]
                { self }
                #[cfg(not(target_endian = "little"))]
                { self.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn from_be(x: $t) -> $t {
                #[cfg(target_endian = "big")]
                { x }
                #[cfg(not(target_endian = "big"))]
                { x.swap_bytes() }
            }
            #[rustc_allow_incoherent_impl]
            pub const fn from_le(x: $t) -> $t {
                #[cfg(target_endian = "little")]
                { x }
                #[cfg(not(target_endian = "little"))]
                { x.swap_bytes() }
            }
        }
    )*)
}

impl_sint_methods! {
    isize, usize;
    i8, u8;
    i16, u16;
    i32, u32;
    i64, u64
}

// region:float_impl
macro_rules! impl_float_methods {
    ($($t:ty, $bits:ty, $nan_bits:expr);*) => ($(
        impl $t {
            pub const NAN: $t = 0.0 / 0.0;
            pub const INFINITY: $t = 1.0 / 0.0;
            pub const NEG_INFINITY: $t = -1.0 / 0.0;

            #[rustc_allow_incoherent_impl]
            pub const fn is_nan(self) -> bool {
                self != self
            }
            #[rustc_allow_incoherent_impl]
            pub const fn is_infinite(self) -> bool {
                // Use abs comparison: self == INFINITY || self == NEG_INFINITY
                (self == <$t>::INFINITY) | (self == <$t>::NEG_INFINITY)
            }
            #[rustc_allow_incoherent_impl]
            pub const fn is_finite(self) -> bool {
                // Finite if not NaN and not infinite
                (self - self) == 0.0
            }
            #[rustc_allow_incoherent_impl]
            pub fn abs(self) -> $t {
                if self < 0.0 { -self } else { self }
            }
            #[rustc_allow_incoherent_impl]
            pub fn min(self, other: $t) -> $t {
                if other.is_nan() || self < other { self } else { other }
            }
            #[rustc_allow_incoherent_impl]
            pub fn max(self, other: $t) -> $t {
                if other.is_nan() || self > other { self } else { other }
            }
            #[rustc_allow_incoherent_impl]
            pub fn clamp(self, min: $t, max: $t) -> $t {
                if self < min {
                    min
                } else if self > max {
                    max
                } else {
                    self
                }
            }
        }
    )*)
}

impl_float_methods! {
    f32, u32, 0x7FC00000u32
}
// endregion:float_impl

impl f32 {
    #[rustc_allow_incoherent_impl]
    pub fn exp(self) -> f32 {
        crate::core::builtins::expf(self)
    }
    #[rustc_allow_incoherent_impl]
    pub fn ln(self) -> f32 {
        crate::core::builtins::logf(self)
    }
    #[rustc_allow_incoherent_impl]
    pub fn sqrt(self) -> f32 {
        crate::core::builtins::sqrtf(self)
    }
    #[rustc_allow_incoherent_impl]
    pub fn floor(self) -> f32 {
        unsafe { crate::core::intrinsics::floorf32(self) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn ceil(self) -> f32 {
        unsafe { crate::core::intrinsics::ceilf32(self) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn round(self) -> f32 {
        unsafe { crate::core::intrinsics::roundf32(self) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn trunc(self) -> f32 {
        unsafe { crate::core::intrinsics::truncf32(self) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn copysign(self, sign: f32) -> f32 {
        unsafe { crate::core::intrinsics::copysignf32(self, sign) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn mul_add(self, a: f32, b: f32) -> f32 {
        unsafe { crate::core::intrinsics::fmaf32(self, a, b) }
    }
    #[rustc_allow_incoherent_impl]
    #[allow(unnecessary_transmutes)]
    pub const fn to_bits(self) -> u32 {
        unsafe { crate::core::mem::transmute(self) }
    }
    #[rustc_allow_incoherent_impl]
    #[allow(unnecessary_transmutes)]
    pub const fn from_bits(bits: u32) -> f32 {
        unsafe { crate::core::mem::transmute(bits) }
    }
    #[rustc_allow_incoherent_impl]
    #[allow(unnecessary_transmutes)]
    pub const fn to_ne_bytes(self) -> [u8; 4] {
        unsafe { crate::core::mem::transmute(self) }
    }
    #[rustc_allow_incoherent_impl]
    pub fn from_ne_bytes(bytes: [u8; 4]) -> f32 {
        unsafe { crate::core::ptr::read(&bytes as *const _ as *const f32) }
    }
}

// endregion:int_impl

// region:error
pub mod error {
    #[rustc_has_incoherent_inherent_impls]
    pub trait Error: crate::core::fmt::Debug + crate::core::fmt::Display {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            None
        }
    }
}
// endregion:error

pub mod tuple {
    // See core/src/primitive_docs.rs for documentation.

    use crate::cmp::Ordering::{self, *};
    use crate::ops::ControlFlow::{self, Break, Continue};

    // Recursive macro for implementing n-ary tuple functions and operations
    //
    // Also provides implementations for tuples with lesser arity. For example, tuple_impls!(A B C)
    // will implement everything for (A, B, C), (A, B) and (A,).
    macro_rules! tuple_impls {
        // Stopping criteria (1-ary tuple)
        ($T:ident) => {
            tuple_impls!(@impl $T);
        };
        // Running criteria (n-ary tuple, with n >= 2)
        ($T:ident $( $U:ident )+) => {
            tuple_impls!($( $U )+);
            tuple_impls!(@impl $T $( $U )+);
        };
        // "Private" internal implementation
        (@impl $( $T:ident )+) => {
            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($T: PartialEq),+> PartialEq for ($($T,)+) {
                    #[inline]
                    fn eq(&self, other: &($($T,)+)) -> bool {
                        $( ${ignore($T)} self.${index()} == other.${index()} )&&+
                    }
                    #[inline]
                    fn ne(&self, other: &($($T,)+)) -> bool {
                        $( ${ignore($T)} self.${index()} != other.${index()} )||+
                    }
                }
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($T: Eq),+> Eq for ($($T,)+)
                {}
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($T: PartialOrd),+> PartialOrd for ($($T,)+)
                {
                    #[inline]
                    fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                        lexical_partial_cmp!($( ${ignore($T)} self.${index()}, other.${index()} ),+)
                    }
                    // lt, le, gt, ge use the correct default impls from PartialOrd
                    // which delegate to partial_cmp
                }
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($T: Ord),+> Ord for ($($T,)+)
                {
                    #[inline]
                    fn cmp(&self, other: &($($T,)+)) -> Ordering {
                        lexical_cmp!($( ${ignore($T)} self.${index()}, other.${index()} ),+)
                    }
                }
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($T: Default),+> Default for ($($T,)+) {
                    #[inline]
                    fn default() -> ($($T,)+) {
                        ($({ let x: $T = Default::default(); x},)+)
                    }
                }
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "array_tuple_conv", since = "1.71.0")]
                // can't do const From due to https://github.com/rust-lang/rust/issues/144280
                impl<T> From<[T; ${count($T)}]> for ($(${ignore($T)} T,)+) {
                    #[inline]
                    #[allow(non_snake_case)]
                    fn from(array: [T; ${count($T)}]) -> Self {
                        let [$($T,)+] = array;
                        ($($T,)+)
                    }
                }
            }

            maybe_tuple_doc! {
                $($T)+ @
                #[stable(feature = "array_tuple_conv", since = "1.71.0")]
                // can't do const From due to https://github.com/rust-lang/rust/issues/144280
                impl<T> From<($(${ignore($T)} T,)+)> for [T; ${count($T)}] {
                    #[inline]
                    #[allow(non_snake_case)]
                    fn from(tuple: ($(${ignore($T)} T,)+)) -> Self {
                        let ($($T,)+) = tuple;
                        [$($T,)+]
                    }
                }
            }
        }
    }

    // If this is a unary tuple, it adds a doc comment.
    // Otherwise, it hides the docs entirely.
    macro_rules! maybe_tuple_doc {
        ($a:ident @ #[$meta:meta] $item:item) => {
            #[doc = "This trait is implemented for tuples up to twelve items long."]
            $item
        };
        ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
            #[doc(hidden)]
            $item
        };
    }

    // Constructs an expression that performs a lexical ordering using method `$rel`.
    // The values are interleaved, so the macro invocation for
    // `(a1, a2, a3) < (b1, b2, b3)` would be `lexical_ord!(lt, opt_is_lt, a1, b1,
    // a2, b2, a3, b3)` (and similarly for `lexical_cmp`)
    //
    // `$chain_rel` is the chaining method from `PartialOrd` to use for all but the
    // final value, to produce better results for simple primitives.
    macro_rules! lexical_ord {
        ($rel: ident, $chain_rel: ident, $a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {{
            match PartialOrd::$chain_rel(&$a, &$b) {
                Break(val) => val,
                Continue(()) => lexical_ord!($rel, $chain_rel, $($rest_a, $rest_b),+),
            }
        }};
        ($rel: ident, $chain_rel: ident, $a:expr, $b:expr) => {
            // Use the specific method for the last element
            PartialOrd::$rel(&$a, &$b)
        };
    }

    // Same parameter interleaving as `lexical_ord` above
    macro_rules! lexical_chain {
        ($chain_rel: ident, $a:expr, $b:expr $(,$rest_a:expr, $rest_b:expr)*) => {{
            PartialOrd::$chain_rel(&$a, &$b)?;
            lexical_chain!($chain_rel $(,$rest_a, $rest_b)*)
        }};
        ($chain_rel: ident) => {
            Continue(())
        };
    }

    macro_rules! lexical_partial_cmp {
        ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
            match ($a).partial_cmp(&$b) {
                Some(Equal) => lexical_partial_cmp!($($rest_a, $rest_b),+),
                ordering => ordering
            }
        };
        ($a:expr, $b:expr) => { ($a).partial_cmp(&$b) };
    }

    macro_rules! lexical_cmp {
        ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
            match ($a).cmp(&$b) {
                Equal => lexical_cmp!($($rest_a, $rest_b),+),
                ordering => ordering
            }
        };
        ($a:expr, $b:expr) => { ($a).cmp(&$b) };
    }

    tuple_impls!(E D C B A Z Y X W V U T);
}

// Software math implementations for NPU (no libm available).
// These provide the linker symbols that bisheng emits when lowering
// llvm.intr.exp/log/sqrt intrinsics.
pub mod builtins {
    #[allow(unnecessary_transmutes)]
    #[inline(always)]
    fn f32_to_bits(x: f32) -> u32 {
        unsafe { crate::core::mem::transmute::<f32, u32>(x) }
    }
    #[allow(unnecessary_transmutes)]
    #[inline(always)]
    fn bits_to_f32(x: u32) -> f32 {
        unsafe { crate::core::mem::transmute::<u32, f32>(x) }
    }

    /// Software expf: exp(x) for f32.
    /// Range reduction + polynomial approximation.
    #[no_mangle]
    pub extern "C" fn expf(x: f32) -> f32 {
        const LN2_HI: f32 = 6.931_457_5e-1; // high bits of ln(2)
        const LN2_LO: f32 = 1.428_606_8e-6; // low bits of ln(2)
        const INV_LN2: f32 = 1.442_695_0; // 1/ln(2)

        let bits = f32_to_bits(x);
        let abs_bits = bits & 0x7FFF_FFFF;

        // NaN passthrough
        if abs_bits > 0x7F80_0000 {
            return x;
        }
        // Overflow
        if x > 88.72 {
            return bits_to_f32(0x7F80_0000); // +inf
        }
        // Underflow
        if x < -87.33 {
            return 0.0;
        }

        // Range reduction: x = n * ln2 + r, |r| <= ln2/2
        let n = if x >= 0.0 {
            (x * INV_LN2 + 0.5) as i32
        } else {
            (x * INV_LN2 - 0.5) as i32
        };
        let nf = n as f32;
        let r = x - nf * LN2_HI - nf * LN2_LO;

        // Polynomial: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
        let r2 = r * r;
        let p = r2 * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * 0.008_333_334)));
        let result = 1.0 + r + p;

        // Scale by 2^n via exponent bit manipulation
        if n < -126 {
            return 0.0;
        }
        if n > 127 {
            return bits_to_f32(0x7F80_0000);
        }
        let scale = bits_to_f32(((n + 127) as u32) << 23);
        result * scale
    }

    /// Software logf: ln(x) for f32.
    /// Decomposes x = 2^e * m, then polynomial on reduced range.
    #[no_mangle]
    pub extern "C" fn logf(x: f32) -> f32 {
        const LN2: f32 = 0.693_147_18;

        let bits = f32_to_bits(x);

        // x <= 0
        if bits >> 31 != 0 {
            return bits_to_f32(0x7FC0_0000); // NaN
        }
        // log(0) = -inf
        if bits == 0 {
            return bits_to_f32(0xFF80_0000); // -inf
        }
        // NaN/inf passthrough
        if bits >= 0x7F80_0000 {
            return x;
        }

        // Decompose: x = 2^e * (1+f) where 1 <= 1+f < 2
        let e = ((bits >> 23) & 0xFF) as i32 - 127;
        let m_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
        let m = bits_to_f32(m_bits); // m in [1, 2)

        // Adjust range: if m > sqrt(2), halve it and add 1 to exponent
        let (m, e) = if m > 1.414_213_6 {
            (m * 0.5, e + 1)
        } else {
            (m, e)
        };

        let f = m - 1.0; // f in [-0.293, 0.414]

        // log(1+f) ≈ f - f²/2 + f³/3 - f⁴/4 + f⁵/5
        let f2 = f * f;
        let p = f - f2 * (0.5 - f * (0.333_333_34 - f * (0.25 - f * 0.2)));

        p + (e as f32) * LN2
    }

    /// Software sqrtf: sqrt(x) for f32.
    /// Bit-manipulation initial guess + Newton-Raphson iterations.
    #[no_mangle]
    pub extern "C" fn sqrtf(x: f32) -> f32 {
        let bits = f32_to_bits(x);

        // Negative → NaN
        if bits >> 31 != 0 && bits != 0x8000_0000 {
            return bits_to_f32(0x7FC0_0000);
        }
        // Zero, inf, NaN passthrough
        if bits == 0 || bits == 0x8000_0000 {
            return x;
        }
        if bits >= 0x7F80_0000 {
            return x;
        }

        // Initial approximation via bit manipulation
        let approx_bits = (bits >> 1) + 0x1FC0_0000;
        let mut y = bits_to_f32(approx_bits);

        // Three Newton-Raphson iterations: y = (y + x/y) / 2
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y
    }
}

// region:column
#[rustc_builtin_macro]
#[macro_export]
macro_rules! column {
    () => {};
}
// endregion:column

pub mod prelude {
    pub mod v1 {
        pub use crate::core::{
            clone::Clone,                                 // :clone
            cmp::Ordering::{Equal, Greater, Less},        // :ordering
            cmp::{Eq, PartialEq},                         // :eq
            cmp::{Ord, PartialOrd},                       // :ord
            convert::AsMut,                               // :as_mut
            convert::AsRef,                               // :as_ref
            convert::{From, Into, TryFrom, TryInto},      // :from
            default::Default,                             // :default
            fmt::derive::Debug,                           // :fmt, derive
            hash::derive::Hash,                           // :hash, derive
            iter::{FromIterator, IntoIterator, Iterator}, // :iterator
            macros::builtin::{derive, derive_const},      // :derive
            marker::Copy,                                 // :copy
            marker::Send,                                 // :send
            marker::Sized,                                // :sized
            marker::Sync,                                 // :sync
            mem::align_of,                                // :align_of
            mem::drop,                                    // :drop
            mem::forget,                                  // :forget
            mem::size_of,                                 // :size_of
            mem::MaybeUninit,                             // :maybe_uninit
            ops::Drop,                                    // :drop
            ops::{AsyncFn, AsyncFnMut, AsyncFnOnce},      // :async_fn
            ops::{Fn, FnMut, FnOnce},                     // :fn
            option::Option::{self, None, Some},           // :option
            panic,                                        // :panic
            result::Result::{self, Err, Ok},              // :result
            str::FromStr,                                 // :str
        };
    }

    pub mod rust_2015 {
        pub use super::v1::*;
    }

    pub mod rust_2018 {
        pub use super::v1::*;
    }

    pub mod rust_2021 {
        pub use super::v1::*;
    }

    pub mod rust_2024 {
        pub use super::v1::*;
    }
}

#[prelude_import]
#[allow(unused)]
pub use prelude::v1::*;
