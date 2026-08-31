use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

#[proc_macro_derive(Interpolate)]
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

#[proc_macro_derive(SimdInterpolate)]
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
