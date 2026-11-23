//! Derive macros for the varying types of `cpu_rasteriser` shaders.
//!
//! Use these derives through `cpu_rasteriser`, which re-exports them alongside the traits they
//! implement as `cpu_rasteriser::{Interpolate, SimdInterpolate}`, and in its prelude. The generated
//! code refers to items of `cpu_rasteriser`, so this crate isn't useful on its own.
//!
//! - `#[derive(Interpolate)]` implements `Interpolate`, which every vertex shader's varyings need.
//! - `#[derive(SimdInterpolate)]` implements `SimdInterpolate`, which the varyings of a
//!   `SimdPipeline` draw also need, and generates the 8-wide type its fragment shader receives.
//!
//! The documentation of the `Interpolate` and `SimdInterpolate` traits in `cpu_rasteriser` has
//! examples of both derives.

#![warn(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

/// Derives `Interpolate` for a varying struct, applying each operation to every field.
///
/// The derive generates two implementations for the struct:
///
/// - `Interpolate`, whose `interpolate`, `difference`, `scale`, and `add_scaled` build a new value
///   by calling the same method on each field;
/// - `Clone`, cloning each field, because `Interpolate` requires it. Don't also derive `Clone`;
///   the two implementations would conflict.
///
/// Use it through `cpu_rasteriser`, which re-exports it as `cpu_rasteriser::Interpolate`. The
/// `Interpolate` trait in `cpu_rasteriser` has an example.
///
/// # Requirements
///
/// These are compile errors:
///
/// - deriving it for an enum or union ("Interpolate can only be derived for structs");
/// - deriving it for a tuple struct ("Interpolate only supports named fields") or a unit struct
///   ("Interpolate cannot be derived for unit structs");
/// - a field whose type doesn't implement `Interpolate` and `Clone`, such as `Colour`;
/// - a struct with generic or lifetime parameters, which the generated implementations don't
///   declare;
/// - `Interpolate` not being imported where the derive is used. The derived impl refers to the
///   trait as plain `Interpolate`; `use cpu_rasteriser::prelude::*` imports it.
///
/// Implement `Interpolate` by hand for types these rule out.
#[proc_macro_derive(Interpolate)]
#[expect(
    clippy::missing_panics_doc,
    reason = "the only `unwrap`s are on the idents of named fields, which always have one"
)]
pub fn derive_interpolate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(name, "Interpolate only supports named fields")
                    .to_compile_error()
                    .into();
            }
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name,
                    "Interpolate cannot be derived for unit structs",
                )
                .to_compile_error()
                .into();
            }
        },

        _ => {
            return syn::Error::new_spanned(name, "Interpolate can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let field_names: Vec<_> = fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap())
        .collect();

    let expanded = quote! {
        impl Interpolate for #name {
            fn interpolate(&self, other: &Self, t: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.interpolate(&other.#field_names, t),
                    )*
                }
            }

            fn difference(&self, other: &Self) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.difference(&other.#field_names),
                    )*
                }
            }

            fn scale(&self, factor: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.scale(factor),
                    )*
                }
            }

            fn add_scaled(&self, other: &Self, factor: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.add_scaled(&other.#field_names, factor),
                    )*
                }
            }
        }

        impl Clone for #name {
            fn clone(&self) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.clone(),
                    )*
                }
            }
        }
    };

    expanded.into()
}

/// Derives `SimdInterpolate` for a varying struct, and generates the 8-wide struct a SIMD fragment
/// shader receives.
///
/// For a struct `Name`, the derive generates:
///
/// - a struct `NameSimd`, deriving `Clone` and `Copy`, with the same visibility as `Name` and one
///   field of the same name for each field of `Name`. An `f32` field becomes an `f32x8`, and a
///   `Vec2`, `Vec3`, or `Vec4` field an array of 2, 3, or 4 `f32x8`, one per component in `x`,
///   `y`, `z`, `w` order. Each `f32x8` holds that component for eight fragments, one per lane,
///   with lanes as `FragmentShaderSimd` maps them to pixels. The fields are private, so a
///   fragment shader that reads them must be in the same module as `Name`.
/// - `impl SimdInterpolate for Name` with `type Simd = NameSimd`, applying each operation
///   component by component.
///
/// Use it through `cpu_rasteriser`, which re-exports it as `cpu_rasteriser::SimdInterpolate`. The
/// `SimdInterpolate` trait in `cpu_rasteriser` has an example.
///
/// # Requirements
///
/// These are compile errors:
///
/// - deriving it for an enum or union ("SimdInterpolate can only be derived for structs");
/// - deriving it for a tuple struct ("SimdInterpolate only supports named fields") or a unit
///   struct ("SimdInterpolate cannot be derived for unit structs");
/// - a field whose type isn't written as a path ending in `f32`, `Vec2`, `Vec3`, or `Vec4`
///   ("SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields"), or isn't a path at all,
///   such as an array or a reference ("SimdInterpolate only supports named types"). Types are
///   matched by the last segment of their name, so an alias with another name is rejected, and a
///   different type with one of these names is accepted but then fails to type-check;
/// - `Name` not implementing `Interpolate`, the supertrait; derive it as well;
/// - a struct with generic or lifetime parameters, which the generated items don't declare;
/// - an existing item named `NameSimd` in the same module;
/// - `SimdInterpolate`, or `Vec2`, `Vec3`, or `Vec4` where a field has that type, not being
///   imported where the derive is used. The derived impl refers to them by their plain names;
///   `use cpu_rasteriser::prelude::*` imports them.
///
/// The generated code refers to `::cpu_rasteriser::wide::f32x8`, so the crate using the derive
/// must depend on `cpu_rasteriser` under that name. Implement `SimdInterpolate` by hand for types
/// these rule out.
#[proc_macro_derive(SimdInterpolate)]
#[expect(
    clippy::missing_panics_doc,
    reason = "the only `unwrap`s are on the idents of named fields, which always have one"
)]
pub fn derive_simd_interpolate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let simd_name = format_ident!("{}Simd", name);
    let visibility = input.vis;

    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(name, "SimdInterpolate only supports named fields")
                    .to_compile_error()
                    .into();
            }
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name,
                    "SimdInterpolate cannot be derived for unit structs",
                )
                .to_compile_error()
                .into();
            }
        },

        _ => {
            return syn::Error::new_spanned(
                name,
                "SimdInterpolate can only be derived for structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let field_names: Vec<_> = fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap())
        .collect();

    let field_types: Vec<_> = fields.iter().map(|field| field.ty.clone()).collect();

    let simd_types = match field_types
        .iter()
        .map(simd_type)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(types) => types,
        Err(error) => return error.to_compile_error().into(),
    };

    let simd_step_fields = match fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            simd_step_expr(&field.ty, field_name)
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(exprs) => exprs,
        Err(error) => return error.to_compile_error().into(),
    };

    let simd_add_scaled_fields = match fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            simd_add_scaled_expr(&field.ty, field_name)
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(exprs) => exprs,
        Err(error) => return error.to_compile_error().into(),
    };

    let simd_perspective_fields = match fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            simd_perspective_expr(&field.ty, field_name)
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(exprs) => exprs,
        Err(error) => return error.to_compile_error().into(),
    };

    let simd_extract_fields = match fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            simd_extract_expr(&field.ty, field_name)
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(exprs) => exprs,
        Err(error) => return error.to_compile_error().into(),
    };

    let expanded = quote! {
        #[derive(Clone, Copy)]
        #visibility struct #simd_name {
            #(
                #field_names: #simd_types,
            )*
        }

        impl SimdInterpolate for #name {
            type Simd = #simd_name;

            #[inline(always)]
            fn simd_step(
                value: &Self,
                step: &Self,
                lanes: ::cpu_rasteriser::wide::f32x8,
            ) -> Self::Simd {
                #simd_name {
                    #(
                        #field_names: #simd_step_fields,
                    )*
                }
            }

            #[inline(always)]
            fn simd_add_scaled(
                value: &Self::Simd,
                step: &Self,
                scale: ::cpu_rasteriser::wide::f32x8,
            ) -> Self::Simd {
                #simd_name {
                    #(
                        #field_names: #simd_add_scaled_fields,
                    )*
                }
            }

            #[inline(always)]
            fn simd_perspective(
                value: Self::Simd,
                perspective: ::cpu_rasteriser::wide::f32x8,
            ) -> Self::Simd {
                #simd_name {
                    #(
                        #field_names: #simd_perspective_fields,
                    )*
                }
            }

            #[inline(always)]
            fn simd_extract_all(
                value: &Self::Simd,
            ) -> [Self; 8] {
                std::array::from_fn(|lane| {
                    Self {
                        #(
                            #field_names: #simd_extract_fields,
                        )*
                    }
                })
            }
        }
    };

    expanded.into()
}

fn simd_type(ty: &Type) -> syn::Result<proc_macro2::TokenStream> {
    match type_name(ty)?.as_str() {
        "f32" => Ok(quote! { ::cpu_rasteriser::wide::f32x8 }),
        "Vec2" => Ok(quote! { [::cpu_rasteriser::wide::f32x8; 2] }),
        "Vec3" => Ok(quote! { [::cpu_rasteriser::wide::f32x8; 3] }),
        "Vec4" => Ok(quote! { [::cpu_rasteriser::wide::f32x8; 4] }),

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields",
        )),
    }
}

fn simd_step_expr(ty: &Type, field: &syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    match type_name(ty)?.as_str() {
        "f32" => Ok(quote! {
            ::cpu_rasteriser::wide::f32x8::splat(value.#field)
                + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field)
        }),

        "Vec2" => Ok(quote! {
            [
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.x)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.x),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.y)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.y),
            ]
        }),

        "Vec3" => Ok(quote! {
            [
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.x)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.x),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.y)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.y),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.z)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.z),
            ]
        }),

        "Vec4" => Ok(quote! {
            [
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.x)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.x),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.y)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.y),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.z)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.z),
                ::cpu_rasteriser::wide::f32x8::splat(value.#field.w)
                    + lanes * ::cpu_rasteriser::wide::f32x8::splat(step.#field.w),
            ]
        }),

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields",
        )),
    }
}

fn simd_add_scaled_expr(ty: &Type, field: &syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    match type_name(ty)?.as_str() {
        "f32" => Ok(quote! {
            value.#field
                + ::cpu_rasteriser::wide::f32x8::splat(step.#field) * scale
        }),

        "Vec2" => Ok(quote! {
            [
                value.#field[0]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.x) * scale,
                value.#field[1]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.y) * scale,
            ]
        }),

        "Vec3" => Ok(quote! {
            [
                value.#field[0]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.x) * scale,
                value.#field[1]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.y) * scale,
                value.#field[2]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.z) * scale,
            ]
        }),

        "Vec4" => Ok(quote! {
            [
                value.#field[0]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.x) * scale,
                value.#field[1]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.y) * scale,
                value.#field[2]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.z) * scale,
                value.#field[3]
                    + ::cpu_rasteriser::wide::f32x8::splat(step.#field.w) * scale,
            ]
        }),

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields",
        )),
    }
}

fn simd_perspective_expr(ty: &Type, field: &syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    match type_name(ty)?.as_str() {
        "f32" => Ok(quote! {
            value.#field * perspective
        }),

        "Vec2" => Ok(quote! {
            [
                value.#field[0] * perspective,
                value.#field[1] * perspective,
            ]
        }),

        "Vec3" => Ok(quote! {
            [
                value.#field[0] * perspective,
                value.#field[1] * perspective,
                value.#field[2] * perspective,
            ]
        }),

        "Vec4" => Ok(quote! {
            [
                value.#field[0] * perspective,
                value.#field[1] * perspective,
                value.#field[2] * perspective,
                value.#field[3] * perspective,
            ]
        }),

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields",
        )),
    }
}

fn simd_extract_expr(ty: &Type, field: &syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    match type_name(ty)?.as_str() {
        "f32" => Ok(quote! {
            value.#field.to_array()[lane]
        }),

        "Vec2" => Ok(quote! {
            Vec2::new(
                value.#field[0].to_array()[lane],
                value.#field[1].to_array()[lane],
            )
        }),

        "Vec3" => Ok(quote! {
            Vec3::new(
                value.#field[0].to_array()[lane],
                value.#field[1].to_array()[lane],
                value.#field[2].to_array()[lane],
            )
        }),

        "Vec4" => Ok(quote! {
            Vec4::new(
                value.#field[0].to_array()[lane],
                value.#field[1].to_array()[lane],
                value.#field[2].to_array()[lane],
                value.#field[3].to_array()[lane],
            )
        }),

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports f32, Vec2, Vec3, and Vec4 fields",
        )),
    }
}

fn type_name(ty: &Type) -> syn::Result<String> {
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            let segment = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(ty, "missing type name"))?;

            Ok(segment.ident.to_string())
        }

        _ => Err(syn::Error::new_spanned(
            ty,
            "SimdInterpolate only supports named types",
        )),
    }
}
