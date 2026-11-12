// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{
    FnArg, GenericArgument, ItemFn, Pat, PatIdent, PathArguments, Type, TypePath,
    parse_macro_input, parse_quote,
};

/// The `#[tile_kernel]` attribute marks a function as an accelerator kernel
/// entry point.
///
/// Two input shapes are supported:
///
/// 1. **Raw-pointer shape (legacy)** — params are `*const T` / `*mut T`.
///    The body is responsible for all unsafe setup: `get_block_idx()`,
///    `GmDeviceCtx::new()`, `ctx.view{,_mut}::<R,C,T>(ptr.wrapping_add(off))`.
///
/// 2. **GmView shape (preferred)** — params are `GmView<'_, R, C, T>` /
///    `GmViewMut<'_, R, C, T>`. The macro:
///      * rewrites the signature back to raw pointers (so the codegen
///        backend sees the same raw-pointer ABI it has always seen —
///        `#[repr(transparent)]` makes this a literal no-op at the ABI
///        level);
///      * emits a prelude at the top of the body that reads
///        `get_block_idx()`, mints a `GmDeviceCtx`, and produces typed
///        `GmView`/`GmViewMut` bindings using the param names the author
///        wrote — `R` and `C` are lifted from the type generics.
///
///    The kernel body is then free of `unsafe` blocks for the boundary
///    plumbing. The only legitimate remaining unsafe is reading a hardware
///    register (if the kernel needs `get_block_idx()` *itself* for other
///    purposes — it does not, because the macro already read it).
///
/// Mixing is allowed: scalar / non-view params pass through untouched.
#[proc_macro_attribute]
pub fn tile_kernel(input: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    expand(input, item)
}

/// Deprecated alias for [`tile_kernel`], kept so any not-yet-migrated
/// `#[aiv_kernel]` call sites still compile. Both entry points delegate to
/// the shared [`expand`] helper.
#[doc(hidden)]
#[deprecated(note = "renamed to `tile_kernel`")]
#[proc_macro_attribute]
pub fn aiv_kernel(input: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    expand(input, item)
}

/// Shared expansion logic for the kernel-entry attribute. Both
/// [`tile_kernel`] (canonical) and the deprecated [`aiv_kernel`] alias call
/// into this.
/// Parsed attribute args for `#[tile_kernel(...)]`.
struct KernelAttrs {
    budget: Option<proc_macro2::TokenStream>,
    k_unroll: Option<proc_macro2::TokenStream>,
}

/// Parse `budget = <expr>` and/or `schedule(k_unroll = <expr>)` from the
/// attribute args. Bare-expr fallback (no `=`, no `schedule(`) stays budget
/// for backwards compatibility with existing kernels.
fn parse_kernel_attrs(input: proc_macro2::TokenStream) -> KernelAttrs {
    let mut out = KernelAttrs {
        budget: None,
        k_unroll: None,
    };
    if input.is_empty() {
        return out;
    }
    let s = input;
    let text = s.to_string();

    // Split on top-level commas so `budget = N, schedule(k_unroll = M)` works.
    let mut parts: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in text.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }

    let mut saw_explicit = false;
    for part in parts {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Some(rest) = p.strip_prefix("budget") {
            saw_explicit = true;
            let expr = rest.trim().trim_start_matches('=').trim();
            if !expr.is_empty() {
                out.budget = expr.parse().ok();
            }
        } else if p.starts_with("schedule") {
            saw_explicit = true;
            // Inside schedule(...): k_unroll = <expr>
            let inner = p
                .trim_start_matches("schedule")
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim();
            for field in inner.split(',') {
                let field = field.trim();
                if let Some(rest) = field.strip_prefix("k_unroll") {
                    let expr = rest.trim().trim_start_matches('=').trim();
                    if !expr.is_empty() {
                        out.k_unroll = expr.parse().ok();
                    }
                }
            }
        } else if let Some(rest) = p.strip_prefix("k_unroll") {
            // Flat form: k_unroll = <expr> (no schedule wrapper).
            saw_explicit = true;
            let expr = rest.trim().trim_start_matches('=').trim();
            if !expr.is_empty() {
                out.k_unroll = expr.parse().ok();
            }
        }
    }

    // Bare-expr legacy: `#[tile_kernel(4096)]` / `#[tile_kernel(budget = 4096)]`
    // already handled above via prefix; if nothing matched, treat whole input
    // as budget when it has no identifiers that look like schedule fields.
    if !saw_explicit && out.budget.is_none() && out.k_unroll.is_none() {
        let expr_str = if let Some(eq) = text.find('=') {
            text[eq + 1..].trim().to_string()
        } else {
            text.trim().to_string()
        };
        out.budget = expr_str.parse().ok();
    }
    out
}

fn expand(input: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    let KernelAttrs { budget, k_unroll } = parse_kernel_attrs(input.into());
    let mut item = parse_macro_input!(item as ItemFn);

    // Walk the signature, pulling any GmView / GmViewMut params out.
    // For each one we:
    //   (a) record (name, R, C, T, is_mut) for the prelude,
    //   (b) rewrite the param in-place to `name_ptr: *const T` / `*mut T`
    //       so the emitted signature is still raw-pointer ABI.
    let mut view_params: Vec<ViewParam> = Vec::new();

    for arg in item.sig.inputs.iter_mut() {
        if let FnArg::Typed(pat_ty) = arg
            && let Some(view) = match_view_type(&pat_ty.ty)
        {
            let orig_name = ident_of_pat(&pat_ty.pat).unwrap_or_else(|| {
                // Should be rare — caller used a tuple/wildcard pattern on a
                // view. We can't bind it then; fall back to a generated name.
                syn::Ident::new("__tile_view", proc_macro2::Span::call_site())
            });
            let ptr_name = syn::Ident::new(&format!("__tile_ptr_{}", orig_name), orig_name.span());

            // Rewrite the param:  GmView<'_, R, C, T>  ->  *const T
            //                     GmViewMut<'_, R, C, T> -> *mut T
            let elem_ty = &view.elem;
            let new_ty: Type = if view.is_mut {
                parse_quote!(*mut #elem_ty)
            } else {
                parse_quote!(*const #elem_ty)
            };
            *pat_ty.ty = new_ty;

            // Rewrite the pattern to the shadow pointer name.
            *pat_ty.pat = Pat::Ident(PatIdent {
                attrs: Vec::new(),
                by_ref: None,
                mutability: None,
                ident: ptr_name.clone(),
                subpat: None,
            });

            view_params.push(ViewParam {
                orig_name,
                ptr_name,
                rows: view.rows,
                cols: view.cols,
                elem: view.elem,
                is_mut: view.is_mut,
            });
        }
    }

    // If any view params were found, inject the boundary prelude at the top
    // of the body. The prelude reads the block index once, mints a ctx, and
    // binds each view using the author's original param name.
    if !view_params.is_empty() {
        let mut prelude = build_prelude(&view_params);
        // F1 capacity guard: if the kernel declared a `budget`, assert each tile's
        // footprint (ROWS * COLS * size_of::<T>()) fits it AT COMPILE TIME. A tile
        // that overflows the target's shared/UB budget fails to compile — the same
        // InBudget check the paper's capacity guard makes, now on the typed source.
        if let Some(budget_expr) = &budget {
            let mut guards: Vec<syn::Stmt> = Vec::new();
            for v in &view_params {
                let (r, c, t, _name) = (&v.rows, &v.cols, &v.elem, v.orig_name.to_string());
                // no_core / const-context capacity check: index a len-1 array at
                // 0 (fits) or 1 (over budget -> out-of-bounds const-eval error). This is
                // panic-free (no_std kernels forbid the panic path a message-assert emits).
                // no_core-safe compile-time capacity check: require the trait bound
                // BudgetCheck<{footprint <= budget}>: BudgetOk. False -> unresolved bound
                // -> type error, NO panic/unwind path (kernels are no_std/no_core).
                let guard: syn::Stmt = parse_quote! {
                    const _: fn() = || {
                        fn _budget_ok()
                        where
                            ::tile_std::BudgetCheck<
                                { (#r) * (#c) * ::tile_std::core::mem::size_of::<#t>() <= (#budget_expr) }
                            >: ::tile_std::BudgetOk,
                        {}
                    };
                };
                guards.push(guard);
            }
            // Guards go first so a budget violation is the first error the author sees.
            guards.extend(prelude);
            prelude = guards;
        }
        // Splice the prelude statements to the front of the block.
        let mut new_stmts: Vec<syn::Stmt> = prelude;
        new_stmts.append(&mut item.block.stmts);
        item.block.stmts = new_stmts;
    }

    // Finally, the two attributes the codegen backend relies on:
    //   * `#[unsafe(no_mangle)]`   — preserve the symbol name
    //   * `#[tile::kernel]`        — codegen marker (see attributes.rs)
    let no_mangle = parse_quote!(#[unsafe(no_mangle)]);
    item.attrs.push(no_mangle);
    let kernel_attr = parse_quote!(#[tile::kernel]);
    item.attrs.push(kernel_attr);

    // If a budget was declared, also emit a `#[tile::budget(N)]` codegen marker so
    // the rustc backend stamps `tile.budget = N` onto the MLIR function. That is the
    // OTHER end of the F1 guard: the emitter then writes a `static_assert` into the
    // generated kernel. One `budget=` on the source thus (a) rejects over-budget
    // source (the trait bound above) AND (b) guards the deployed kernel.
    if let Some(budget_expr) = &budget {
        let budget_attr: syn::Attribute = parse_quote!(#[tile::budget(#budget_expr)]);
        item.attrs.push(budget_attr);
    }

    // If a schedule was declared (`schedule(k_unroll = N)`), emit the flat
    // `#[tile::k_unroll(N)]` marker so the rustc backend stamps
    // `tile.k_unroll = N` onto the MLIR function. Emitters that lower matmul
    // through emit_unrolled_k_accumulation honour it (same sum order).
    if let Some(k_expr) = &k_unroll {
        let k_attr: syn::Attribute = parse_quote!(#[tile::k_unroll(#k_expr)]);
        item.attrs.push(k_attr);
    }

    item.into_token_stream().into()
}

/// Parsed info about a `GmView<'a, R, C, T>` / `GmViewMut<'a, R, C, T>` type.
struct ViewMatch {
    rows: proc_macro2::TokenStream,
    cols: proc_macro2::TokenStream,
    elem: Type,
    is_mut: bool,
}

/// Info we keep per view-typed param so we can emit the prelude.
struct ViewParam {
    orig_name: syn::Ident,
    ptr_name: syn::Ident,
    rows: proc_macro2::TokenStream,
    cols: proc_macro2::TokenStream,
    elem: Type,
    is_mut: bool,
}

/// Extract `(R, C, T, is_mut)` from a type if it's `GmView<'_, R, C, T>` or
/// `GmViewMut<'_, R, C, T>`. Matches both bare and path-qualified forms.
fn match_view_type(ty: &Type) -> Option<ViewMatch> {
    let path = match ty {
        Type::Path(TypePath { path, .. }) => path,
        _ => return None,
    };

    // The last segment of the path holds the generic args.
    let seg = path.segments.last()?;
    let is_mut = match seg.ident.to_string().as_str() {
        "GmView" => false,
        "GmViewMut" => true,
        _ => return None,
    };

    let generics = match &seg.arguments {
        PathArguments::AngleBracketed(g) => &g.args,
        _ => return None,
    };

    // Expect `<'a, R, C, T>`: 1 lifetime + 2 consts + 1 type = 4 args.
    // Skip the lifetime; take R, C, T in order.
    let mut iter = generics
        .iter()
        .filter(|g| !matches!(g, GenericArgument::Lifetime(_)));
    let rows_arg = iter.next()?;
    let cols_arg = iter.next()?;
    let elem_arg = iter.next()?;
    // Accept trailing args (future-compat) but ignore them.

    let rows = match rows_arg {
        GenericArgument::Const(e) => e.to_token_stream(),
        GenericArgument::Type(t) => t.to_token_stream(),
        _ => return None,
    };
    let cols = match cols_arg {
        GenericArgument::Const(e) => e.to_token_stream(),
        GenericArgument::Type(t) => t.to_token_stream(),
        _ => return None,
    };
    let elem = match elem_arg {
        GenericArgument::Type(t) => t.clone(),
        _ => return None,
    };

    Some(ViewMatch {
        rows,
        cols,
        elem,
        is_mut,
    })
}

fn ident_of_pat(pat: &Pat) -> Option<syn::Ident> {
    if let Pat::Ident(PatIdent { ident, .. }) = pat {
        Some(ident.clone())
    } else {
        None
    }
}

/// Build the boundary prelude that the macro splices at the top of the
/// kernel body. The prelude:
///
///   * reads the block index once,
///   * mints one `GmDeviceCtx`,
///   * for each view-typed param, offsets the raw pointer by
///     `block_idx * R * C` and mints a typed view under the author's
///     original name.
///
/// All unsafe is confined to this prelude; the author's body runs with
/// pure safe bindings.
fn build_prelude(views: &[ViewParam]) -> Vec<syn::Stmt> {
    let mut stmts: Vec<syn::Stmt> = Vec::new();

    let read_block: syn::Stmt = parse_quote! {
        let __tile_block_idx: usize = unsafe { ::tile_std::get_block_idx() } as usize;
    };
    stmts.push(read_block);

    let mint_ctx: syn::Stmt = parse_quote! {
        let __tile_ctx = unsafe { ::tile_std::tile::GmDeviceCtx::new() };
    };
    stmts.push(mint_ctx);

    for v in views {
        let orig = &v.orig_name;
        let ptr = &v.ptr_name;
        let r = &v.rows;
        let c = &v.cols;
        let t = &v.elem;

        // Each view gets its own offset: block_idx * R * C.
        let stmt: syn::Stmt = if v.is_mut {
            parse_quote! {
                let #orig: ::tile_std::tile::GmViewMut<'_, #r, #c, #t> = unsafe {
                    __tile_ctx.view_mut::<#r, #c, #t>(
                        #ptr.wrapping_add(__tile_block_idx * #r * #c)
                    )
                };
            }
        } else {
            parse_quote! {
                let #orig: ::tile_std::tile::GmView<'_, #r, #c, #t> = unsafe {
                    __tile_ctx.view::<#r, #c, #t>(
                        #ptr.wrapping_add(__tile_block_idx * #r * #c)
                    )
                };
            }
        };
        stmts.push(stmt);
    }

    // Suppress the `unused` warning on the ctx when the body never touches
    // it directly (it's typed so that lifetimes still work).
    let shut_up: syn::Stmt = parse_quote! {
        let _ = &__tile_ctx;
    };
    stmts.push(shut_up);

    stmts
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    // NOTE: the `tile_kernel` entry takes a `proc_macro::TokenStream`, which can
    // only be constructed inside an actual macro expansion — so it is covered by
    // the integration build wherever `tile_std` kernels compile, not here. The
    // crate's REAL logic lives in the `syn`/`proc_macro2`-typed helpers below,
    // which take `syn::Type` / `syn::Pat` and so ARE unit-testable off-toolchain
    // (no rustc/trybuild fixture needed).

    fn ty(s: &str) -> Type {
        syn::parse_str::<Type>(s).expect("parse type")
    }

    #[test]
    fn match_view_type_immutable() {
        let m = match_view_type(&ty("GmView<'a, 32, 64, f32>")).expect("GmView matches");
        assert!(!m.is_mut, "GmView must be immutable");
        assert_eq!(m.rows.to_string(), "32");
        assert_eq!(m.cols.to_string(), "64");
        assert_eq!(m.elem.to_token_stream().to_string(), "f32");
    }

    #[test]
    fn match_view_type_mutable() {
        let m = match_view_type(&ty("GmViewMut<'a, 8, 16, i32>")).expect("GmViewMut matches");
        assert!(m.is_mut, "GmViewMut must be mutable");
        assert_eq!(m.rows.to_string(), "8");
        assert_eq!(m.cols.to_string(), "16");
    }

    #[test]
    fn match_view_type_path_qualified() {
        // The last path segment carries the generics, so a fully-qualified
        // path still matches.
        let m = match_view_type(&ty("::tile_std::tile::GmView<'_, 4, 4, f32>"))
            .expect("qualified GmView matches");
        assert!(!m.is_mut);
        assert_eq!(m.rows.to_string(), "4");
    }

    #[test]
    fn match_view_type_rejects_non_view() {
        assert!(
            match_view_type(&ty("*const f32")).is_none(),
            "raw ptr is not a view"
        );
        assert!(
            match_view_type(&ty("u32")).is_none(),
            "scalar is not a view"
        );
        assert!(
            match_view_type(&ty("Vec<f32>")).is_none(),
            "Vec is not a view"
        );
        // A GmView with too few generics (missing the element type) is rejected.
        assert!(
            match_view_type(&ty("GmView<'a, 32>")).is_none(),
            "incomplete view rejected"
        );
    }

    fn attrs(s: &str) -> KernelAttrs {
        let ts: proc_macro2::TokenStream = s.parse().expect("parse attr args");
        parse_kernel_attrs(ts)
    }

    #[test]
    fn parse_budget_only() {
        let a = attrs("budget = 4096");
        assert_eq!(a.budget.map(|t| t.to_string()), Some("4096".into()));
        assert!(a.k_unroll.is_none());
    }

    #[test]
    fn parse_schedule_k_unroll_only() {
        let a = attrs("schedule(k_unroll = 8)");
        assert!(a.budget.is_none());
        assert_eq!(a.k_unroll.map(|t| t.to_string()), Some("8".into()));
    }

    #[test]
    fn parse_budget_and_schedule_together() {
        let a = attrs("budget = 4096, schedule(k_unroll = 16)");
        assert_eq!(a.budget.map(|t| t.to_string()), Some("4096".into()));
        assert_eq!(a.k_unroll.map(|t| t.to_string()), Some("16".into()));
    }

    #[test]
    fn parse_empty_args() {
        let a = attrs("");
        assert!(a.budget.is_none());
        assert!(a.k_unroll.is_none());
    }

    #[test]
    fn parse_flat_k_unroll_without_schedule_wrapper() {
        let a = attrs("k_unroll = 4");
        assert!(a.budget.is_none());
        assert_eq!(a.k_unroll.map(|t| t.to_string()), Some("4".into()));
    }

    /// Extract the binding pattern from a parsed `fn` arg (`syn::Pat` has no
    /// direct `Parse` impl, so we route through a full function signature).
    fn pat_of(arg: &str) -> Pat {
        let f: ItemFn =
            syn::parse_str::<ItemFn>(&format!("fn k({arg}) {{}}")).expect("parse fn for pat");
        match f.sig.inputs.into_iter().next().expect("one arg") {
            FnArg::Typed(pt) => *pt.pat,
            _ => panic!("not a typed arg"),
        }
    }

    #[test]
    fn ident_of_pat_binds_simple_ident() {
        assert_eq!(
            ident_of_pat(&pat_of("foo: u32")).map(|i| i.to_string()),
            Some("foo".to_string())
        );
        // A wildcard pattern has no ident.
        assert!(
            ident_of_pat(&pat_of("_: u32")).is_none(),
            "wildcard has no ident"
        );
    }

    #[test]
    fn build_prelude_emits_boundary_plumbing() {
        let v = ViewParam {
            orig_name: syn::Ident::new("x", proc_macro2::Span::call_site()),
            ptr_name: syn::Ident::new("__tile_ptr_x", proc_macro2::Span::call_site()),
            rows: quote!(32),
            cols: quote!(64),
            elem: ty("f32"),
            is_mut: false,
        };
        let stmts = build_prelude(&[v]);
        let rendered = quote!(#(#stmts)*).to_string();
        // reads the block index once
        assert!(
            rendered.contains("get_block_idx"),
            "prelude must read block idx:\n{rendered}"
        );
        // mints exactly one ctx
        assert!(
            rendered.contains("GmDeviceCtx :: new") || rendered.contains("GmDeviceCtx::new"),
            "prelude must mint a ctx:\n{rendered}"
        );
        // immutable view binds via .view::<...>
        assert!(
            rendered.contains("view ::") || rendered.contains("view::"),
            "immutable view must use .view:\n{rendered}"
        );
        // offset is block_idx * R * C
        assert!(
            rendered.contains("wrapping_add"),
            "prelude must offset the ptr:\n{rendered}"
        );
    }

    #[test]
    fn build_prelude_mutable_uses_view_mut() {
        let v = ViewParam {
            orig_name: syn::Ident::new("y", proc_macro2::Span::call_site()),
            ptr_name: syn::Ident::new("__tile_ptr_y", proc_macro2::Span::call_site()),
            rows: quote!(8),
            cols: quote!(8),
            elem: ty("i32"),
            is_mut: true,
        };
        let stmts = build_prelude(&[v]);
        let rendered = quote!(#(#stmts)*).to_string();
        assert!(
            rendered.contains("view_mut") || rendered.contains("view_mut ::"),
            "mutable view must use .view_mut:\n{rendered}"
        );
    }
}
